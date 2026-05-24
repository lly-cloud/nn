import sys
import glob
import os

import carla
import random
import time
import pygame
import numpy as np
import math
from ultralytics import YOLO
import torch

# Initialize Pygame for display
def init_pygame(width, height):
    pygame.init()
    display = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Driver's View")
    return display

# Convert CARLA image to numpy array (RGB)
def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]  # Drop alpha
    return array

# Load YOLOv8 pretrained model for traffic sign detection
model = YOLO("yolov8n.pt")  # Use yolov8n.pt for fast inference

# Run detection on RGB numpy image from CARLA camera
def detect_traffic_signs(image_np):
    results = model.predict(source=image_np, imgsz=640, conf=0.5, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False)
    detections = results[0].boxes.data.cpu().numpy()
    names = results[0].names

    signs_detected = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = names[int(cls)]
        signs_detected.append((label, conf, (int(x1), int(y1), int(x2), int(y2))))
    return signs_detected

# Calculate the steering angle between vehicle and waypoint
def get_steering_angle(vehicle_transform, waypoint_transform):
    v_loc = vehicle_transform.location
    v_forward = vehicle_transform.get_forward_vector()
    wp_loc = waypoint_transform.location
    direction = wp_loc - v_loc
    direction = carla.Vector3D(direction.x, direction.y, 0.0)

    v_forward = carla.Vector3D(v_forward.x, v_forward.y, 0.0)
    norm_dir = math.sqrt(direction.x ** 2 + direction.y ** 2)
    norm_fwd = math.sqrt(v_forward.x ** 2 + v_forward.y ** 2)
    dot = v_forward.x * direction.x + v_forward.y * direction.y
    angle = math.acos(dot / (norm_dir * norm_fwd + 1e-5))
    cross = v_forward.x * direction.y - v_forward.y * direction.x
    if cross < 0:
        angle *= -1
    return angle

# Control based on traffic signs
def control_vehicle_based_on_sign(vehicle, detected_signs, simulation_time):
    velocity = vehicle.get_velocity()
    current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6

    # Traffic light control
    traffic_light_state = vehicle.get_traffic_light_state()
    if traffic_light_state == carla.TrafficLightState.Red:
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        return

    for sign, conf, _ in detected_signs:
        if "stop" in sign.lower() and conf > 0.5:
            control = carla.VehicleControl()
            control.brake = 1.0
            vehicle.apply_control(control)
            time.sleep(2)
        elif "speed limit" in sign.lower():
            digits = [int(s) for s in sign.split() if s.isdigit()]
            if digits:
                speed_limit = digits[0]
                if current_speed < speed_limit:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0))
                else:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.5))

# Spawn traffic signs
def spawn_dynamic_elements(world, blueprint_library):
    spawn_points = world.get_map().get_spawn_points()
    signs = []
    speed_values = [20, 40, 60, 60, 40, 60, 40, 20]
    sign_bp = [bp for bp in blueprint_library if 'static.prop.speedlimit' in bp.id or 'static.prop.stop' in bp.id]

    for i, speed in enumerate(speed_values):
        for bp in sign_bp:
            if f"speedlimit.{speed}" in bp.id:
                transform = spawn_points[i % len(spawn_points)]
                transform.location.z = 0
                actor = world.try_spawn_actor(bp, transform)
                if actor:
                    signs.append(actor)
                break

    stop_signs = [bp for bp in blueprint_library if 'static.prop.stop' in bp.id]
    if stop_signs:
        transform = spawn_points[-1]
        transform.location.z = 0
        actor = world.try_spawn_actor(stop_signs[0], transform)
        if actor:
            signs.append(actor)
    return signs

# Main function
def main():
    actor_list = []
    max_speed = 0
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        elements = spawn_dynamic_elements(world, blueprint_library)
        actor_list.extend(elements)

        # Spawn ego vehicle
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_point = random.choice(map.get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        # Spawn NPC traffic
        for _ in range(10):
            traffic_bp = random.choice(blueprint_library.filter('vehicle.*'))
            traffic_spawn = random.choice(map.get_spawn_points())
            npc = world.try_spawn_actor(traffic_bp, traffic_spawn)
            if npc:
                npc.set_autopilot(True)
                actor_list.append(npc)

        # Camera
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        display = init_pygame(800, 600)
        image_surface = [None]

        def image_callback(image):
            image_surface[0] = process_image(image)

        camera.listen(image_callback)

        # Top-down view
        spectator = world.get_spectator()
        def update_spectator():
            t = vehicle.get_transform()
            spectator.set_transform(carla.Transform(t.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

        clock = pygame.time.Clock()
        start_time = time.time()

        while True:
            update_spectator()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            # Steering control
            trans = vehicle.get_transform()
            waypoint = map.get_waypoint(trans.location)
            next_wp = waypoint.next(2.0)[0]
            angle = get_steering_angle(trans, next_wp.transform)
            steer = max(-1.0, min(1.0, angle * 2.0))

            control = carla.VehicleControl()
            control.throttle = 0.5
            control.steer = steer
            control.brake = 0.0
            vehicle.apply_control(control)

            # Speed & time
            elapsed = time.time() - start_time
            m = int(elapsed // 60)
            s = int(elapsed % 60)
            vel = vehicle.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6
            if speed > max_speed:
                max_speed = round(speed, 1)

            if image_surface[0] is not None:
                signs = detect_traffic_signs(image_surface[0])
                control_vehicle_based_on_sign(vehicle, signs, elapsed)

                surf = pygame.image.frombuffer(image_surface[0].tobytes(), (800, 600), "RGB")
                display.blit(surf, (0, 0))

                font = pygame.font.Font(None, 26)
                display.blit(font.render(f"Time: {m:02d}:{s:02d}", True, (0,255,0)), (10,10))
                display.blit(font.render(f"Speed: {speed:.1f} km/h", True, (255,255,0)), (10,40))
                display.blit(font.render(f"Max: {max_speed} km/h", True, (255,0,0)), (10,70))

                pygame.display.flip()

            clock.tick_busy_loop(30)
            if elapsed > 120:
                break

    finally:
        for actor in actor_list:
            actor.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()