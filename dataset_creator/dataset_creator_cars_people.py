#!/usr/bin/env python
import glob
import os
import cv2
import sys

import argparse
import yaml
import time

from CarlaSyncMode import CarlaSyncMode
from coco_creator import CocoCreator

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='preview images in window')
    parser.add_argument('--lights', action='store_true', help='collect traffic lights')
    parser.add_argument('--pedestrians', action='store_true', help='collect pedestrians')
    parser.add_argument('--cars', action='store_true', help='collect cars')
    
    args = parser.parse_args()
    
    use_gui = args.gui
    use_lights = args.lights
    use_pedestrians = args.pedestrians
    use_cars = args.cars
    
    classes = {
        'pedestrian': 1,
        'car': 2,
        'light': 3,
    }
    
    cc = CocoCreator(classes=classes)
    img_cnt = 0

    actor_list = []
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.load_world('Town01')

    
    if use_gui:
        window_name = 'Preview'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())

        blueprint_library = world.get_blueprint_library()
        
        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.audi.*')),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)
        vehicle.set_autopilot(True)
        
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(0)
        traffic_manager.ignore_lights_percentage(vehicle, 100)

        camera = blueprint_library.find('sensor.camera.rgb')
        image_size_x = camera.get_attribute('image_size_x').as_int()
        image_size_y = camera.get_attribute('image_size_y').as_int()
        FOV = camera.get_attribute('fov').as_float()

        camera_rgb = world.spawn_actor(
            camera,
            carla.Transform(carla.Location(x=2.5, z=1.6), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        

        camera_seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')        
        camera_seg = world.spawn_actor(
            camera_seg_bp,
            carla.Transform(carla.Location(x=2.5, z=1.6), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_seg)
        
        K = build_projection_matrix(image_size_x, image_size_y, FOV)
        
        lights_bounding_box = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
            
        # Create a synchronous mode context.
        list_actor = world.get_actors()
        
        color_cny = np.array([0, 0, 0])
        color_map = {
            'red': 0,
            'yellow': 1,
            'green': 2,
        }
        
        state_map = {
            0: carla.TrafficLightState.Red,
            1: carla.TrafficLightState.Yellow,
            2: carla.TrafficLightState.Green,
        }
        
                # print(actor_.get_light_boxes())
        start = time.time()
        frame_cnt = 0
        with CarlaSyncMode(world, camera_rgb, camera_seg, fps=1) as sync_mode:
            while True:  
                if frame_cnt % 10 == 0:
                
                    # Advance the simulation and wait for the data.
                    snapshot, image_rgb, image_seg  = sync_mode.tick(timeout=2.0)
                    array = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8"))
                    img = np.reshape(array, (image_rgb.height, image_rgb.width, 4))
                    
                    array = np.frombuffer(image_seg.raw_data, dtype=np.dtype("uint8"))
                    img_seg = np.reshape(array, (image_seg.height, image_seg.width, 4))

                    found_detections = False
                    
                    
                    if use_lights:
                        world_2_camera = np.array(camera_rgb.get_transform().get_inverse_matrix())
                        seg_mask = (img_seg[:,:,2] == 18).astype(np.uint8)
                        
                        nearby_bboxes = []
                        for actor_ in list_actor:
                            if isinstance(actor_, carla.TrafficLight):
                                if 4 < actor_.get_transform().location.distance(vehicle.get_transform().location) < 20:
                                    for bbox in lights_bounding_box:
                                        if bbox.location.distance(actor_.get_transform().location) < 2.6:
                                            orientation_trafficsign = bbox.rotation.yaw
                                            if orientation_trafficsign > 180:
                                                orientation_trafficsign = orientation_trafficsign - 360
                                                
                                            veh_rot = vehicle.get_transform().rotation.yaw + 90
                                            
                                            if np.abs(veh_rot - orientation_trafficsign) < 45:
                                                forward_vec = vehicle.get_transform().get_forward_vector()
                                                ray = bbox.location - vehicle.get_transform().location
                                                
                                                nearby_bboxes.append(bbox)
                                                if forward_vec.dot(ray) > 1:
                                                    verts = [v for v in bbox.get_world_vertices(carla.Transform())]
                                                    x_max = -10000
                                                    x_min = 10000
                                                    y_max = -10000
                                                    y_min = 10000

                                                    for vert in verts:
                                                        p = get_image_point(vert, K, world_2_camera)
                                                        # Find the rightmost vertex
                                                        if p[0] > x_max:
                                                            x_max = p[0]
                                                        # Find the leftmost vertex
                                                        if p[0] < x_min:
                                                            x_min = p[0]
                                                        # Find the highest vertex
                                                        if p[1] > y_max:
                                                            y_max = p[1]
                                                        # Find the lowest  vertex
                                                        if p[1] < y_min:
                                                            y_min = p[1]
                                                            
                                                    x_len = x_max - x_min
                                                    y_len = y_max - y_min
                                                    idx = np.argmin(color_cny)
                                                    actor_.set_state(state_map[idx]) 
                                                    state = actor_.get_state()
                                                    color_cny[color_map[str(state).lower()]] += 1
                                                    
                                                    image_area = img.shape[0]*img.shape[1]
                                                    bbox_area = x_len * y_len
                                                    light_mask = np.zeros(seg_mask.shape)
                                                    light_mask[int(y_min):int(y_min+y_len), int(x_min):int(x_min+x_len)] = seg_mask[int(y_min):int(y_min+y_len), int(x_min):int(x_min+x_len)]
                                                    print(light_mask.shape)
                                                    if 0.05*image_area < bbox_area < 0.5*image_area:
                                                        try:
                                                            cc.annonate_image(img_cnt, [x_min, y_min, x_len, y_len], 'light', mask=seg_mask)
                                                            found_detections = True
                                                        except Exception as e:
                                                            pass
                    
                    
                    if use_pedestrians:
                        seg_mask = (img_seg[:,:,2] == 4).astype(np.uint8)
                        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg_mask)
                        for i, stat in enumerate(stats):
                            if stat[4] < 300*400 and stat[4] > 15*15:
                                seg = np.zeros(seg_mask.shape)
                                seg[stat[1]:stat[1]+stat[3],stat[0]:stat[0]+stat[2]] += seg_mask[stat[1]:stat[1]+stat[3],stat[0]:stat[0]+stat[2]]
                                try:
                                    cc.annonate_image(img_cnt, None, 'pedestrian', mask=seg)
                                    found_detections = True
                                except Exception as e:
                                    pass
                                
                    
                    if use_cars:
                        seg_mask = (img_seg[:,:,2] == 10).astype(np.uint8)
                        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg_mask)
                        for i, stat in enumerate(stats):
                            if stat[4] < 600*600 and stat[4] > 25*25 and centroids[i][0] > 350 and centroids[i][0] < 450:
                                seg = np.zeros(seg_mask.shape)
                                seg[stat[1]:stat[1]+stat[3],stat[0]:stat[0]+stat[2]] += seg_mask[stat[1]:stat[1]+stat[3],stat[0]:stat[0]+stat[2]]
                                try:
                                    cc.annonate_image(img_cnt, None, 'car', mask=seg)
                                    found_detections = True
                                except Exception as e:
                                    pass
                                
                            elif stat[4] < 700*600 and stat[4] > 200*200 and centroids[i][0] > 250 and centroids[i][0] < 650:
                                seg = np.zeros(seg_mask.shape)
                                seg[stat[1]:stat[1]+stat[3],stat[0]:stat[0]+stat[2]] += seg_mask[stat[1]:stat[1]+stat[3],stat[0]:stat[0]+stat[2]]
                                try:
                                    cc.annonate_image(img_cnt, None, 'car', mask=seg)
                                    found_detections = True
                                except Exception as e:
                                    pass

                    
                    if found_detections:
                        cc.add_image(img, img_cnt)
                        img_cnt += 1
                    
                    end = time.time()
                    fps = round(1.0 / (end - start))
                    start = end
                    if use_gui:
                        cv2.putText(img, f'{fps}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                        cv2.imshow(window_name, img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        
                elif frame_cnt % 30 == 0:
                    print(f'In simulation FPS: {fps}')
                    
                frame_cnt += 1

    except Exception as e:
        print(f'Exception: {e}')

    finally:
        if use_gui:
            cv2.destroyAllWindows()
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        print('actors destroyed.')
        cc.dump_json()
        print('dataset saved')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('\nCancelled by user. Bye!')

