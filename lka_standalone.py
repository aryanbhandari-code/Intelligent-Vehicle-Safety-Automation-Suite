
import carla
import numpy as np
import cv2
import math
import time
import queue
import random

# --- TrafficManager Class (Same as in other modules) ---
class TrafficManager:
    """Manages spawning and cleanup of all NPC traffic vehicles."""
    def __init__(self, world, vehicle_id, max_vehicles=10):
        self.world = world
        self.max_vehicles = max_vehicles
        self.traffic_vehicles = []
        self.actor_list = []
        self.blueprint_library = world.get_blueprint_library()
        self.vehicle_id = vehicle_id # For logging
        
    def spawn_traffic(self, ego_vehicle, min_distance=20.0, max_distance=80.0):
        spawn_points = self.world.get_map().get_spawn_points()
        ego_location = ego_vehicle.get_transform().location
        
        valid_spawn_points = [sp for sp in spawn_points if sp.location.distance(ego_location) > min_distance]
        random.shuffle(valid_spawn_points)
        num_to_spawn = min(self.max_vehicles - len(self.traffic_vehicles), len(valid_spawn_points))
        
        for i in range(num_to_spawn):
            spawn_point = valid_spawn_points[i]
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
        print(f"🧹 [{self.vehicle_id}] Cleaning up traffic vehicles...")
        for vehicle in self.actor_list:
            if vehicle.is_alive:
                vehicle.destroy()
        self.traffic_vehicles.clear()
        self.actor_list.clear()

# --- LaneDetector (Perception Class) ---
class LaneDetector:
    """Processes camera images to find lane deviation."""
    def __init__(self, width, height):
        self.IMAGE_WIDTH = width
        self.IMAGE_HEIGHT = height
        self.roi_vertices = np.array([
            (0, self.IMAGE_HEIGHT), 
            (self.IMAGE_WIDTH * 0.45, self.IMAGE_HEIGHT * 0.6), 
            (self.IMAGE_WIDTH * 0.55, self.IMAGE_HEIGHT * 0.6), 
            (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        ], dtype=np.int32)
        
        self.DEVIATION_THRESHOLD = 35 # Pixels from center

    def average_slope_intercept(self, lines):
        """Averages all detected line segments."""
        left_fit = []
        right_fit = []
        if lines is None: return None, None
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x2 - x1 == 0: continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if slope < -0.5: left_fit.append((slope, intercept))
            elif slope > 0.5: right_fit.append((slope, intercept))
        left_fit_avg = np.average(left_fit, axis=0) if len(left_fit) > 0 else None
        right_fit_avg = np.average(right_fit, axis=0) if len(right_fit) > 0 else None
        return left_fit_avg, right_fit_avg

    def draw_lane_lines(self, img, line_params):
        """Extrapolates and draws the averaged lane line."""
        try:
            slope, intercept = line_params
        except TypeError:
            return None
        y1 = self.IMAGE_HEIGHT
        y2 = int(self.IMAGE_HEIGHT * 0.6)
        if slope == 0: return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return x1

    def process_image(self, image_data):
        """Processes the image and returns deviation and a plotted image."""
        img = np.array(image_data.raw_data)
        img = img.reshape((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 4))
        img = img[:, :, :3]
        original_image = np.copy(img)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        
        mask = np.zeros_like(canny)
        cv2.fillPoly(mask, [self.roi_vertices], 255)
        masked_edges = cv2.bitwise_and(canny, mask)
        
        lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 50, np.array([]), 40, 100)
        left_line_params, right_line_params = self.average_slope_intercept(lines)
        
        line_image = np.zeros_like(original_image)
        car_center_x = self.IMAGE_WIDTH / 2
        
        left_x_bottom = self.draw_lane_lines(line_image, left_line_params)
        right_x_bottom = self.draw_lane_lines(line_image, right_line_params)

        if left_x_bottom is not None and right_x_bottom is not None:
            lane_center_x = (left_x_bottom + right_x_bottom) / 2
            deviation = car_center_x - lane_center_x
            
            cv2.line(line_image, (int(lane_center_x), self.IMAGE_HEIGHT), (int(lane_center_x), self.IMAGE_HEIGHT - 50), (0, 255, 0), 3) # Lane center
            cv2.line(line_image, (int(car_center_x), self.IMAGE_HEIGHT), (int(car_center_x), self.IMAGE_HEIGHT - 50), (0, 0, 255), 3) # Car center

          
            if abs(deviation) > self.DEVIATION_THRESHOLD:
                cv2.putText(original_image, "WARNING: LANE DEPARTURE!", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        final_image = cv2.addWeighted(original_image, 0.8, line_image, 1, 0)
        
        # We return the processed image for the GUI window
        return final_image

# --- Main Simulation Class ---
class LaneDepartureWarningSystem:
    """Main class to manage the LDW simulation."""
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.actor_list = []
        self.original_settings = None
        self.image_queue = queue.Queue()
        self.processed_image = None
        
        self.IMAGE_WIDTH = 800
        self.IMAGE_HEIGHT = 600
        
        # --- Helper Classes ---
        self.lane_detector = LaneDetector(self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        self.traffic_manager = None
        self.last_traffic_spawn = 0

    def on_collision(self, event):
        """Collision callback."""
        actor_type = event.other_actor.type_id if event.other_actor else "unknown"
        print(f"COLLISION DETECTED with {actor_type}!")
        
    def main_loop(self):
        """Main simulation loop."""
        spectator = self.world.get_spectator()
        
        while True:
            self.world.tick()
            now = time.time()

            if (now - self.last_traffic_spawn) > 15.0:
                self.traffic_manager.spawn_traffic(self.vehicle)
                self.actor_list.extend(self.traffic_manager.actor_list)
                self.last_traffic_spawn = now

            # Get sensor data
            try:
                image_data = self.image_queue.get(timeout=1.0)
            except queue.Empty:
                print("Warning: Camera queue is empty. Skipping frame.")
                continue
            
            # --- 1. Perception ---
            # Process the image to get the visual output
            self.processed_image = self.lane_detector.process_image(image_data)
            
            # --- 2. Control ---
            # MOVEMENT FIX: No manual control. Autopilot is handling it.
            # self.vehicle.apply_control(...) is removed.

            # --- 3. Visualization ---
            if self.processed_image is not None:
                current_speed_ms = get_speed(self.vehicle)
                cv2.putText(self.processed_image, f"Speed: {current_speed_ms*3.6:.1f} km/h", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(self.processed_image, "MODE: AUTOPILOT", (50, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Lane Departure Warning (LDW)', self.processed_image)
                if cv2.waitKey(1) == ord('q'):
                    break
            
            # Update spectator
            ego_tf = self.vehicle.get_transform()
            cam_loc = ego_tf.location + ego_tf.get_forward_vector() * -10.0 + carla.Location(z=6.0)
            cam_rot = carla.Rotation(pitch=-25, yaw=ego_tf.rotation.yaw)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

    def run(self):
        """Initializes and runs the simulation."""
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.original_settings = self.world.get_settings()
            
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            
            blueprint_library = self.world.get_blueprint_library()
            
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            spawn_point = self.world.get_map().get_spawn_points()[50]
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.actor_list.append(self.vehicle)
            
           
            self.vehicle.set_autopilot(True)
            
            # --- Spawn Traffic Manager ---
            self.traffic_manager = TrafficManager(self.world, "ldw_vehicle") # Give it a name for logging
            self.actor_list.extend(self.traffic_manager.actor_list)
            
            # --- Spawn Sensors ---
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.IMAGE_WIDTH))
            camera_bp.set_attribute('image_size_y', str(self.IMAGE_HEIGHT))
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            self.actor_list.append(self.camera)
            self.camera.listen(self.image_queue.put)
            
            collision_bp = blueprint_library.find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
            self.actor_list.append(self.collision_sensor)
            self.collision_sensor.listen(self.on_collision)
            
            print("LDW System Initialized. Starting simulation...")
            print("   Car is on AUTOPILOT. Watch for warnings when it drifts.")
            print("Press 'q' in the HUD window to stop.\n")
            self.main_loop()

        except KeyboardInterrupt:
            print('\nScript interrupted. Cleaning up...')
        except Exception as e:
            print(f'An error occurred: {e}')
        finally:
            self.cleanup()
            
    def cleanup(self):
        print('Cleaning up actors...')
        if self.world and self.original_settings:
            self.world.apply_settings(self.original_settings)
        
        if self.camera: self.camera.stop()
        if self.traffic_manager: self.traffic_manager.cleanup()
                
        for actor in self.actor_list:
            if actor.is_alive: actor.destroy()
        cv2.destroyAllWindows()
        print('LDW Module Done.')

def get_speed(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

if __name__ == '__main__':
    ldw_system = LaneDepartureWarningSystem()
    ldw_system.run()