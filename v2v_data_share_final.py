

import carla
import paho.mqtt.client as mqtt
import time
import json
import argparse
import math
import random
import queue
import numpy as np
import cv2

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
            if vehicle and vehicle.is_alive: # Added check for `vehicle` being not None
                vehicle.destroy()
        self.traffic_vehicles.clear()
        self.actor_list.clear()

# --- MQTT Configuration ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

class V2VDataSharer:
    def __init__(self, vehicle_id, follow_id=None):
        self.vehicle_id = vehicle_id
        self.other_vehicle_id = "car2" if vehicle_id == "car1" else "car1"
      
        self.spawn_point_index = 20 if vehicle_id == "car1" else 40 
        
        
        self.follow_camera = (follow_id == self.vehicle_id)
        
        self.carla_client = None
        self.world = None
        self.vehicle = None
        self.actor_list = []
        self.original_settings = None
        self.last_message_time = 0
        self.other_car_data = None # This will store the received data
        
        self.camera = None
        self.image_queue = queue.Queue()
        
        self.traffic_manager = None
        self.last_traffic_spawn = 0
        
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=self.vehicle_id)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[MQTT-{self.vehicle_id}] Connected to broker.")
            subscribe_topic = f"v2v/{self.other_vehicle_id}/data"
            client.subscribe(subscribe_topic)
            print(f"[MQTT-{self.vehicle_id}] Subscribed to {subscribe_topic}")
        else:
            print(f"[MQTT-{self.vehicle_id}] Connection failed.")

    def on_message(self, client, userdata, msg):
        """This is the core data-sharing logic. Store the received data."""
        self.last_message_time = time.time()
        self.other_car_data = json.loads(msg.payload.decode())

    def connect_to_carla(self):
        print(f"[{self.vehicle_id}] Attempting to connect to CARLA...")
        self.carla_client = carla.Client('localhost', 2000)
        self.carla_client.set_timeout(10.0)
        self.world = self.carla_client.get_world()
        self.original_settings = self.world.get_settings()
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        print(f"[{self.vehicle_id}] Successfully connected in Sync Mode.")

    def setup_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        spawn_points = self.world.get_map().get_spawn_points()
        if len(spawn_points) <= self.spawn_point_index:
            print(f"Warning: Spawn point index {self.spawn_point_index} is out of bounds. Using a random point.")
            self.spawn_point = random.choice(spawn_points)
        else:
            self.spawn_point = spawn_points[self.spawn_point_index]
        
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, self.spawn_point)
        if self.vehicle is None:
            print(f"[{self.vehicle_id}] Failed to spawn vehicle. Is the spot clear?")
            return False
            
        self.actor_list.append(self.vehicle)
        print(f"[{self.vehicle_id}] Spawned at {self.spawn_point.location}")
        
        # Use AUTOPILOT for robust movement
        self.vehicle.set_autopilot(True)
        
        # Spawn camera for this car's GUI
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(self.image_queue.put)
        
        self.traffic_manager = TrafficManager(self.world, self.vehicle_id)
        # Ensure we capture actors spawned by TrafficManager directly into self.actor_list if needed
        # self.actor_list.extend(self.traffic_manager.actor_list) # This line might duplicate actors, better to let TrafficManager manage its own actors
        
        return True

    def run_simulation(self):
        """Main loop: drive, publish data, and display received data."""
        publish_topic = f"v2v/{self.vehicle_id}/data"
        spectator = self.world.get_spectator()
        
        while True:
            self.world.tick()
            now = time.time()
            
            if not self.vehicle or not self.vehicle.is_alive: break
            
            if (now - self.last_traffic_spawn) > 15.0:
                self.traffic_manager.spawn_traffic(self.vehicle, min_distance=50.0) 
                # Actors spawned by traffic manager are managed internally, no need to extend actor_list here
                self.last_traffic_spawn = now
            
            if self.follow_camera:
                tf = self.vehicle.get_transform()
                spectator.set_transform(carla.Transform(
                    tf.location + tf.get_forward_vector() * -10.0 + carla.Location(z=5.0), 
                    carla.Rotation(pitch=-20, yaw=tf.rotation.yaw)
                ))

            # --- 1. Get My Data (NEW: More detailed) ---
            current_transform = self.vehicle.get_transform()
            current_velocity = self.vehicle.get_velocity()
            current_accel = self.vehicle.get_acceleration()
            current_ang_vel = self.vehicle.get_angular_velocity() # in deg/s

            current_speed_ms = math.sqrt(current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2)
            current_speed_kmh = current_speed_ms * 3.6
            current_heading_deg = current_transform.rotation.yaw
            current_turn_rate_radps = math.radians(current_ang_vel.z) # Convert deg/s to rad/s
            
            # --- 2. Publish My Data (NEW: Richer payload) ---
            payload = {
                "vehicle_id": self.vehicle_id,
                "timestamp": time.time(),
                "kinematics": {
                    "location": {"x": round(current_transform.location.x, 2), "y": round(current_transform.location.y, 2)},
                    "speed_kmh": round(current_speed_kmh, 2),
                    "acceleration_mps2": {"x": round(current_accel.x, 2), "y": round(current_accel.y, 2)},
                    "heading_deg": round(current_heading_deg, 2),
                    "turn_rate_radps": {"z": round(current_turn_rate_radps, 3)}
                }
            }
            self.mqtt_client.publish(publish_topic, json.dumps(payload), qos=0)

            # --- 3. Display GUI ---
            try:
                image_data = self.image_queue.get(timeout=1.0)
                img = np.array(image_data.raw_data)
                img = img.reshape((600, 800, 4))
                hud_image = img[:, :, :3].copy()
            except queue.Empty:
                continue
                
            # Autopilot should always be on
            self.vehicle.set_autopilot(True)
                
            # --- Draw HUD (This part is just for visualization) ---
            
            # Draw My Status
            cv2.putText(hud_image, f"MY STATUS ({self.vehicle_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(hud_image, f"Speed: {current_speed_kmh:.1f} km/h", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(hud_image, f"Accel: {current_accel.x:.1f}, {current_accel.y:.1f} m/s2", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(hud_image, f"Heading: {current_heading_deg:.1f} deg", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw Other Car's Status (This is the data transfer visualization)
            cv2.putText(hud_image, f"RECEIVED ({self.other_vehicle_id})", (430, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if self.other_car_data and (time.time() - self.last_message_time < 2.0):
                # Safely parse the new kinematics object
                kinematics = self.other_car_data.get('kinematics', {})
                other_speed = kinematics.get('speed_kmh', 'N/A')
                other_loc = kinematics.get('location', {'x': '?', 'y': '?'})
                other_accel = kinematics.get('acceleration_mps2', {'x': '?', 'y': '?'})
                other_heading = kinematics.get('heading_deg', 'N/A')
                other_turn_rate_z = kinematics.get('turn_rate_radps', {}).get('z', '?')

                cv2.putText(hud_image, "V2V LINK: OK", (430, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(hud_image, f"Speed: {other_speed} km/h", (430, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(hud_image, f"Loc: {other_loc['x']}, {other_loc['y']}", (430, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(hud_image, f"Accel: {other_accel['x']}, {other_accel['y']}", (430, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(hud_image, f"Heading: {other_heading} deg", (430, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(hud_image, f"Turn Rate: {other_turn_rate_z} rad/s", (430, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(hud_image, "V2V LINK: DOWN", (430, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow(f"V2V Data Monitor ({self.vehicle_id})", hud_image)
            if cv2.waitKey(1) == ord('q'):
                break

    def start(self):
        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            self.connect_to_carla()
            if not self.setup_vehicle():
                return
            self.run_simulation()
        except KeyboardInterrupt:
            print(f"\n[{self.vehicle_id}] Simulation stopped by user.")
        except Exception as e:
            print(f"[{self.vehicle_id}] A critical error occurred: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleans up all actors and restores settings."""
        print(f"\nCleaning up actors for {self.vehicle_id}...")
        if self.world and self.original_settings:
            self.world.apply_settings(self.original_settings)
        
        if self.camera and self.camera.is_listening:
            self.camera.stop()
            
        if self.traffic_manager:
            self.traffic_manager.cleanup()
            
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        for actor in list(self.actor_list):
            if actor and actor.is_alive:
                actor.destroy()
        cv2.destroyAllWindows()
        print(f"[{self.vehicle_id}] Cleanup complete.")

def get_speed(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a V2V Data Sharing Demo.")
    parser.add_argument('--role', required=True, choices=['car1', 'car2'], help="Role of the vehicle.")
    parser.add_argument('--follow', action='store_true', help="Set camera to follow this vehicle.")
    args = parser.parse_args()
    
    follow_id = args.role if args.follow else None
    
   
    simulator = V2VDataSharer(vehicle_id=args.role, follow_id=follow_id)
    
    simulator.start()