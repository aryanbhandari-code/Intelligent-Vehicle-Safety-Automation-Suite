

import carla
import time
import math
import random

# --- PID Controller Class ---
class PIDController:
    """A simple PID controller."""
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0
        self.integral = 0.0

    def control(self, error, dt):
        """Calculates the control output."""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output
    
    def reset(self):
        """Resets the controller state."""
        self.prev_error = 0.0
        self.integral = 0.0

class TrafficManager:
    """Manages spawning and cleanup of all NPC traffic vehicles."""
    def __init__(self, world, max_vehicles=15):
        self.world = world
        self.max_vehicles = max_vehicles
        self.traffic_vehicles = []
        self.actor_list = [] # Keep track of all spawned actors
        self.blueprint_library = world.get_blueprint_library()
        
    def spawn_traffic(self, ego_vehicle, min_distance=20.0, max_distance=80.0):
        """Spawns traffic vehicles around the ego vehicle."""
        spawn_points = self.world.get_map().get_spawn_points()
        ego_location = ego_vehicle.get_transform().location
        
        valid_spawn_points = []
        for sp in spawn_points:
            distance = sp.location.distance(ego_location)
            if min_distance <= distance <= max_distance:
                valid_spawn_points.append(sp)
        
        random.shuffle(valid_spawn_points)
        num_to_spawn = min(self.max_vehicles - len(self.traffic_vehicles), len(valid_spawn_points))
        
        for i in range(num_to_spawn):
            if len(self.traffic_vehicles) >= self.max_vehicles:
                break
                
            spawn_point = valid_spawn_points[i]
            
            vehicle_bps = self.blueprint_library.filter('vehicle.*')
            car_bps = [bp for bp in vehicle_bps if 'bicycle' not in bp.id and 'motorcycle' not in bp.id]
            
            if not car_bps:
                continue
                
            vehicle_bp = random.choice(car_bps)
            
            try:
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if vehicle:
                    vehicle.set_autopilot(True)
                    self.traffic_vehicles.append(vehicle)
                    self.actor_list.append(vehicle)
            except Exception as e:
                print(f"Failed to spawn traffic vehicle: {e}")
    
    def cleanup(self):
        """Cleans up all traffic vehicles."""
        print(" Cleaning up traffic vehicles...")
        for vehicle in self.actor_list:
            if vehicle.is_alive:
                vehicle.destroy()
        self.traffic_vehicles.clear()
        self.actor_list.clear()

class AdaptiveCruiseController:
    """Implements the core ACC logic."""
    def __init__(self, max_speed_kmh=70.0):
        self.max_speed_ms = max_speed_kmh / 3.6
        self.time_gap = 2.5
        self.min_gap = 8.0
        self.emergency_brake = False
        
    def calculate_target_speed(self, lead_vehicle_info, current_speed):
        """Calculates the target speed based on the lead vehicle."""
        target_speed = self.max_speed_ms
        self.emergency_brake = False
        
        if lead_vehicle_info:
            lead_speed = lead_vehicle_info['actor_speed']
            distance_to_lead = lead_vehicle_info['forward_distance']
            
            desired_gap = max(self.min_gap, current_speed * self.time_gap)
            
            if distance_to_lead < desired_gap * 0.7:
                target_speed = max(lead_speed - 3.0, 0.0)
                self.emergency_brake = (distance_to_lead < self.min_gap)
            elif distance_to_lead < desired_gap:
                target_speed = max(lead_speed - 1.0, 0.0)
            elif distance_to_lead > desired_gap * 1.5:
                target_speed = min(lead_speed + 2.0, self.max_speed_ms)
            else:
                target_speed = min(lead_speed + 0.5, self.max_speed_ms)
        
        return target_speed
    
    def should_emergency_stop(self, immediate_threat):
        """Determines if an immediate emergency stop is needed."""
        if not immediate_threat:
            return False
        if (immediate_threat['forward_distance'] < 5.0 or 
            (0 < immediate_threat['time_to_collision'] < 1.5)):
            return True
        return False

class CollisionDetector:
    """Uses CARLA's 'ground truth' to detect obstacles."""
    def __init__(self, world, ego_vehicle):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.collision_sensor = None
        self.has_collided = False
        self._setup_collision_sensor()
        
    def _setup_collision_sensor(self):
        """Initializes the collision sensor."""
        blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(self._on_collision)
    
    def _on_collision(self, event):
        """Callback for collision events."""
        self.has_collided = True
        actor_type = event.other_actor.type_id if event.other_actor else "unknown"
        print(f" COLLISION DETECTED with {actor_type}!")
        
    def get_obstacles_ahead(self, max_distance=60.0, angle_threshold=50.0):
        """Finds all actors in front of the ego vehicle."""
        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_fwd = ego_tf.get_forward_vector()
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        
        obstacles = []
        all_actors = self.world.get_actors()
        
        for actor in all_actors.filter('vehicle.*'): # Only check vehicles
            if actor.id == self.ego_vehicle.id:
                continue
            
            actor_loc = actor.get_transform().location
            distance = actor_loc.distance(ego_loc)
            
            if distance > max_distance:
                continue
            
            # Check if actor is in front
            dx = actor_loc.x - ego_loc.x
            dy = actor_loc.y - ego_loc.y
            angle_to_actor = math.atan2(dy, dx)
            angle_diff = abs(angle_to_actor - ego_yaw)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
            
            if math.degrees(angle_diff) > angle_threshold:
                continue
            
            # Calculate relative speed and TTC
            ego_speed = get_speed(self.ego_vehicle)
            actor_speed = get_speed(actor)
            relative_speed = ego_speed - actor_speed
            time_to_collision = distance / relative_speed if relative_speed > 0.5 else float('inf')
            
            obstacles.append({
                'actor': actor,
                'forward_distance': distance * math.cos(angle_diff),
                'lateral_distance': abs(distance * math.sin(angle_diff)),
                'relative_speed': relative_speed,
                'time_to_collision': time_to_collision,
                'actor_speed': actor_speed,
            })
        
        obstacles.sort(key=lambda x: x['forward_distance'])
        return obstacles
    
    def get_immediate_threat(self, max_distance=25.0):
        """Finds the most immediate threat."""
        obstacles = self.get_obstacles_ahead(max_distance)
        for obstacle in obstacles:
            if (obstacle['forward_distance'] < 10.0 or 
                (0 < obstacle['time_to_collision'] < 2.5)):
                return obstacle
        return None
    
    def get_lead_vehicle(self, max_distance=50.0):
        """Finds the vehicle directly in the current lane."""
        obstacles = self.get_obstacles_ahead(max_distance, angle_threshold=35.0)
        for obstacle in obstacles:
            if obstacle['lateral_distance'] < 2.5 and obstacle['forward_distance'] > 0:
                return obstacle
        return None

# --- Utility Functions ---
def get_speed(vehicle):
    """Returns vehicle speed in m/s"""
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

def calculate_steering_angle(vehicle_transform, target_waypoint):
    """Calculates the steering angle to reach the target waypoint."""
    target_location = target_waypoint.transform.location
    dx = target_location.x - vehicle_transform.location.x
    dy = target_location.y - vehicle_transform.location.y
    
    target_angle = math.atan2(dy, dx)
    vehicle_angle = math.radians(vehicle_transform.rotation.yaw)
    
    angle_diff = target_angle - vehicle_angle
    angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
    return angle_diff

def main():
    """Main function to run the ACC simulation."""
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = None
    original_settings = None
    actor_list = [] # List to store all spawned actors for cleanup
    
    try:
        world = client.get_world()
        original_settings = world.get_settings()
        blueprint_library = world.get_blueprint_library()
        world_map = world.get_map()

        # --- 1. Setup Environment (Synchronous Mode) ---
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 # 20 FPS
        world.apply_settings(settings)

        # --- 2. Spawn Ego Vehicle ---
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
        spawn_point = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if not ego_vehicle:
            raise Exception("Failed to spawn ego vehicle.")
        actor_list.append(ego_vehicle)
        print(" Ego vehicle spawned successfully.")

        # --- 3. Spawn Traffic ---
        traffic_manager = TrafficManager(world, max_vehicles=8)
        traffic_manager.spawn_traffic(ego_vehicle, min_distance=30.0, max_distance=70.0)
        actor_list.extend(traffic_manager.traffic_vehicles)
        print(f" {len(traffic_manager.traffic_vehicles)} traffic vehicles spawned")

        # --- 4. Initialize Controllers ---
        speed_pid = PIDController(Kp=0.4, Ki=0.01, Kd=0.1)
        steer_pid = PIDController(Kp=0.7, Ki=0.0, Kd=0.15)
        collision_detector = CollisionDetector(world, ego_vehicle)
        actor_list.append(collision_detector.collision_sensor)
        
        acc_controller = AdaptiveCruiseController(max_speed_kmh=70.0)

        # --- 5. Simulation Parameters ---
        absolute_max_speed_kmh = 70.0
        absolute_max_speed_ms = absolute_max_speed_kmh / 3.6
        
        spectator = world.get_spectator()
        last_traffic_spawn_time = time.time()
        start_time = time.time()
        
        print("\n ADAPTIVE CRUISE CONTROL SYSTEM (Ground Truth)")
        print("   (Overtaking removed for focus)")
        print("Press Ctrl+C to stop.\n")

        # --- 6. Main Simulation Loop ---
        while True:
            world.tick()
            now = time.time()
            dt = now - start_time
            start_time = now
            if dt < 0.001: dt = 0.05 # Avoid dt=0

            # --- Safety Check ---
            if collision_detector.has_collided:
                print(" COLLISION! Stopping simulation.")
                break

            # --- Traffic Management ---
            if (now - last_traffic_spawn_time) > 25.0:
                traffic_manager.spawn_traffic(ego_vehicle, min_distance=35.0, max_distance=80.0)
                actor_list.extend(traffic_manager.traffic_vehicles)
                last_traffic_spawn_time = now

            # --- Perception (Ground Truth) ---
            ego_speed = get_speed(ego_vehicle)
            lead_vehicle_info = collision_detector.get_lead_vehicle(max_distance=45.0)
            immediate_threat = collision_detector.get_immediate_threat(max_distance=20.0)

            # --- Planning ---
            # A. Determine Driving Mode
            driving_mode = "FOLLOWING" if lead_vehicle_info else "CRUISING"
            
            # B. Get Target Waypoint (Simple path following)
            target_waypoint = world_map.get_waypoint(ego_vehicle.get_location()).next(5.0)[0]

            # C. Get Target Speed
            target_speed = acc_controller.calculate_target_speed(lead_vehicle_info, ego_speed)
            target_speed = min(target_speed, absolute_max_speed_ms)
            
            # D. Check for Emergency Stop
            if acc_controller.should_emergency_stop(immediate_threat):
                target_speed = 0.0
                print(" EMERGENCY BRAKING (Threat Detected)!")
            
            # --- Control ---
            # A. Steering Control
            steer = steer_pid.control(calculate_steering_angle(ego_vehicle.get_transform(), target_waypoint), dt)
            steer = max(-1.0, min(steer, 1.0))

            # B. Speed Control
            speed_error = target_speed - ego_speed
            throttle_brake = speed_pid.control(speed_error, dt)

            if target_speed <= 0.0:
                throttle = 0.0
                brake = 1.0
            elif throttle_brake > 0:
                throttle = max(0.0, min(throttle_brake, 0.6)) # Gentle acceleration
                brake = 0.0
            else:
                throttle = 0.0
                brake = max(0.0, min(abs(throttle_brake), 1.0))

            # --- Apply Control ---
            ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

            # --- Update Spectator ---
            ego_tf = ego_vehicle.get_transform()
            cam_loc = ego_tf.location + ego_tf.get_forward_vector() * -12.0 + carla.Location(z=5.0)
            cam_rot = carla.Rotation(pitch=-15, yaw=ego_tf.rotation.yaw)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # --- Print Status ---
            print(f"Mode: {driving_mode} | Speed: {ego_speed*3.6:.1f} km/h -> {target_speed*3.6:.1f} km/h | T:{throttle:.2f} B:{brake:.2f} S:{steer:.2f}", end='\r')

    except KeyboardInterrupt:
        print("\n Simulation stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\n🧹 Cleaning up...")
        if world and original_settings:
            original_settings.synchronous_mode = False
            world.apply_settings(original_settings)
        
        if 'traffic_manager' in locals():
            traffic_manager.cleanup()
            
        for actor in actor_list:
            if actor and actor.is_alive:
                actor.destroy()
        
        print("Cleanup complete.")

if __name__ == '__main__':
    main()
