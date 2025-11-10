

import carla
import numpy as np
import cv2
from ultralytics import YOLO
import time
import queue
import math
import random

# --- TrafficManager (Same as in ACC) ---
class TrafficManager:
    """Manages spawning and cleanup of all NPC traffic vehicles."""
    def __init__(self, world, max_vehicles=10):
        self.world = world
        self.max_vehicles = max_vehicles
        self.traffic_vehicles = []
        self.actor_list = []
        self.blueprint_library = world.get_blueprint_library()
        
    def spawn_traffic(self, ego_vehicle):
        """Spawns traffic in front of the ego vehicle."""
        ego_tf = ego_vehicle.get_transform()
        ego_wp = self.world.get_map().get_waypoint(ego_tf.location)
        
        spawn_points = []
        for i in range(15, 50, 10): # Spawn cars 15m to 50m ahead
            wp = ego_wp.next(i)[0]
            if wp:
                spawn_points.append(wp.transform)
        
        num_to_spawn = min(self.max_vehicles - len(self.traffic_vehicles), len(spawn_points))
        random.shuffle(spawn_points)
        
        for i in range(num_to_spawn):
            spawn_point = spawn_points[i]
            spawn_point.location.z += 1.0 # Avoid spawning underground
            
            vehicle_bps = self.blueprint_library.filter('vehicle.*')
            car_bps = [bp for bp in vehicle_bps if 'bicycle' not in bp.id and 'motorcycle' not in bp.id]
            if not car_bps: continue
            
            vehicle_bp = random.choice(car_bps)
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                self.traffic_vehicles.append(vehicle)
                self.actor_list.append(vehicle)
    
    def cleanup(self):
        print("🧹 Cleaning up traffic vehicles...")
        for vehicle in self.actor_list:
            if vehicle.is_alive:
                vehicle.destroy()
        self.traffic_vehicles.clear()
        self.actor_list.clear()

class FusionPerception:
    """Handles all sensor processing (YOLO + Depth)."""
    def __init__(self, width, height, model_path='yolov5s.pt'):
        self.IMAGE_WIDTH = width
        self.IMAGE_HEIGHT = height
        print("Loading YOLOv5 model...")
        self.model = YOLO(model_path)
        self.model.to('cuda')
        self.class_names = self.model.names
        print("YOLOv5 model loaded successfully.")
        
    def process_rgb(self, image_data):
        """Processes RGB image, runs YOLO, and returns plotted image and detections."""
        img = np.array(image_data.raw_data)
        img = img.reshape((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 4))
        img = img[:, :, :3]

        results = self.model(img, verbose=False) 
        detections = results[0].boxes.data.cpu().numpy()
        plotted_image = results[0].plot()
        return plotted_image, detections

    def process_depth(self, image_data):
        """Processes Depth image and converts to a 2D distance array."""
        img = np.array(image_data.raw_data)
        img = img.reshape((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 4))
        img = img[:, :, :3]
        # Convert BGR-encoded depth to a 2D array of distances in meters
        gray_depth = (img[:, :, 2] + img[:, :, 1] * 256 + img[:, :, 0] * 256 * 256) / (256 * 256 * 256 - 1)
        return gray_depth * 1000.0 # Return distance in meters

    def find_threats(self, detections, depth_image, plotted_image):
        """Fuses detections with depth map to find true distances."""
        threats = []
        for *box, conf, cls_id in detections:
            class_name = self.class_names[int(cls_id)]
            
            if class_name in ['car', 'truck', 'person', 'bicycle'] and conf > 0.6:
                # Get center of 2D box
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                
                # Get 3D distance from depth map
                distance = depth_image[cy, cx]
                
                # Add distance text to the plotted image
                cv2.putText(plotted_image, f"{distance:.1f}m", (cx - 20, cy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                threats.append({'class': class_name, 'distance': distance})
        
        return threats, plotted_image

class FusionAebSystem:
    """Main class to manage the AEB simulation."""
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.actor_list = []
        self.original_settings = None
        
        # --- Sensor Queues for synchronous mode ---
        self.rgb_queue = queue.Queue()
        self.depth_queue = queue.Queue()
        
        self.IMAGE_WIDTH = 800
        self.IMAGE_HEIGHT = 600
        self.AEB_THRESHOLD_METERS = 8.0 # Brake if an object is closer than 8 meters
        
        # --- Helper Classes ---
        self.perception = FusionPerception(self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        self.traffic_manager = None
        self.last_traffic_spawn = 0

    def on_collision(self, event):
        """Collision callback."""
        actor_type = event.other_actor.type_id if event.other_actor else "unknown"
        print(f" COLLISION DETECTED with {actor_type}!")
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

    def main_loop(self):
        """Main simulation loop."""
        spectator = self.world.get_spectator()
        
        while True:
            self.world.tick()
            
            # Spawn traffic periodically
            if (time.time() - self.last_traffic_spawn) > 10.0:
                self.traffic_manager.spawn_traffic(self.vehicle)
                self.actor_list.extend(self.traffic_manager.actor_list)
                self.last_traffic_spawn = time.time()

            # Get data from sensors
            try:
                rgb_data = self.rgb_queue.get(timeout=1.0)
                depth_data = self.depth_queue.get(timeout=1.0)
            except queue.Empty:
                print("Sensor queue timeout. Skipping frame.")
                continue
            
            # --- 1. Perception ---
            plotted_image, detections = self.perception.process_rgb(rgb_data)
            depth_image = self.perception.process_depth(depth_data)
            threats, plotted_image = self.perception.find_threats(detections, depth_image, plotted_image)

            # --- 2. Planning & Control ---
            brake_applied = False
            for threat in threats:
                if threat['distance'] < self.AEB_THRESHOLD_METERS:
                    print(f"!!! 3D-AEB TRIGGER !!! Close {threat['class']} detected at {threat['distance']:.1f}m.")
                    self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
                    brake_applied = True
                    break # Only brake for the closest threat
            
            if not brake_applied:
                # Re-enable autopilot if brake is not applied
                self.vehicle.set_autopilot(True)
            
            # --- 3. Visualization ---
            cv2.imshow('AEB Sensor Fusion (YOLO + Depth)', plotted_image)
            if cv2.waitKey(1) == ord('q'):
                break
            
            # Update spectator to follow the car
            tf = self.vehicle.get_transform()
            cam_loc = tf.location + tf.get_forward_vector() * -10.0 + carla.Location(z=6.0)
            cam_rot = carla.Rotation(pitch=-25, yaw=tf.rotation.yaw)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

    def run(self):
        """Initializes and runs the simulation."""
        try:
            # --- 1. Connect to CARLA & Set Sync Mode ---
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.original_settings = self.world.get_settings()
            
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            
            blueprint_library = self.world.get_blueprint_library()
            
            # --- 2. Spawn Vehicle ---
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            spawn_point = self.world.get_map().get_spawn_points()[0]
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.actor_list.append(self.vehicle)
            self.vehicle.set_autopilot(True) # Autopilot will drive, AEB will override
            
            # --- 3. Spawn Traffic Manager ---
            self.traffic_manager = TrafficManager(self.world)
            self.actor_list.extend(self.traffic_manager.actor_list)
            
            # --- 4. Spawn Sensors (RGB + Depth + Collision) ---
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.IMAGE_WIDTH))
            camera_bp.set_attribute('image_size_y', str(self.IMAGE_HEIGHT))
            rgb_cam = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            self.actor_list.append(rgb_cam)
            
            depth_bp = blueprint_library.find('sensor.camera.depth')
            depth_bp.set_attribute('image_size_x', str(self.IMAGE_WIDTH))
            depth_bp.set_attribute('image_size_y', str(self.IMAGE_HEIGHT))
            depth_cam = self.world.spawn_actor(depth_bp, camera_transform, attach_to=self.vehicle)
            self.actor_list.append(depth_cam)
            
            collision_bp = blueprint_library.find('sensor.other.collision')
            col_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
            self.actor_list.append(col_sensor)
            
            # Listen to sensors
            rgb_cam.listen(self.rgb_queue.put)
            depth_cam.listen(self.depth_queue.put)
            col_sensor.listen(self.on_collision)
            
            print("AEB Fusion System Initialized. Starting simulation...")
            self.main_loop()

        except KeyboardInterrupt:
            print('\nScript interrupted. Cleaning up...')
        except Exception as e:
            print(f'An error occurred: {e}')
        finally:
            print(' Cleaning up actors...')
            if self.world and self.original_settings:
                self.world.apply_settings(self.original_settings)
            
            if 'rgb_cam' in locals() and rgb_cam.is_listening: rgb_cam.stop()
            if 'depth_cam' in locals() and depth_cam.is_listening: depth_cam.stop()
                
            if self.traffic_manager: self.traffic_manager.cleanup()
                
            for actor in self.actor_list:
                if actor.is_alive: actor.destroy()
            cv2.destroyAllWindows()
            print('AEB Module Done.')

if __name__ == '__main__':
    aeb_system = FusionAebSystem()
    aeb_system.run()