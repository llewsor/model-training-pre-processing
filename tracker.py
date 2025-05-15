import math
import os

record_folder_name = "data_trajectory_history"

if not os.path.exists(record_folder_name):
    os.makedirs(record_folder_name)

record_file_location = record_folder_name + "//trajectoryRecord.txt"
file = open(record_file_location,"w")
file.write("ID \t x \t y \t w \t h \tAREA\n-------\t-------\t-------\t-------\t-------\t-------\n")
file.close()

class EuclideanDistTracker:
    def __init__(self):
        # Storing the positions of center of the objects
        self.center_points = {}
        # Count of ID of boundng boxes
        # each time new object will be captured the id will be increassed by 1
        self.id_count = 0
        self.trajectories = {}  # Store past positions for each object
        
    def update(self, objects_rect, fps):
        objects_bbs_ids = []
        
        # Calculating the center of objects
        for rect in objects_rect:
            # print(f'\nForeach rectangle {rect}')
            x, y, w, h, area = rect
            center_x = (x + x + w) // 2
            center_y = (y + y + h) // 2
            # Find if object is already detected or not
            same_object_detected = False
            
            for id, pt in self.center_points.items():
                # print(f'id: {id}, pt: {pt}, center_x: {center_x}, center_y: {center_y}, area : {area}')

                dist = math.hypot(center_x - pt[0], center_y - pt[1])
                # print(f'Distance: {dist}')
                
                # print(self.center_points)
                if fps < 31 and dist < 25 or fps < 121 and  dist < 100:
                    self.center_points[id] = (center_x, center_y)
                    objects_bbs_ids.append([x, y, w, h, area, id])     
                    same_object_detected = True

                    # Store trajectory points
                    if id in self.trajectories:
                        self.trajectories[id].append((center_x, center_y))
                    else:
                        self.trajectories[id] = [(center_x, center_y)]
                    break

            # Assign the ID to the detected object
            if same_object_detected is False:
                self.center_points[self.id_count] = (center_x, center_y)                      
                objects_bbs_ids.append([x, y, w, h, area, self.id_count])
                self.trajectories[self.id_count] = [(center_x, center_y)]  # Start tracking trajectory
                self.id_count += 1
                
        # Cleaning the dictionary ids that are not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, object_id = obj_bb_id
            center = self.center_points.get(object_id)
            if center is not None:
                new_center_points[object_id] = center
            
        # Updating the dictionary with IDs that is not used
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

    def capture(self, x, y, h, w, id, area):
        filet = open(record_file_location, "a")
        filet.write(f"{id} \t {x} \t {y} \t {str(w)} \t {str(h)} \t {area}\n")
        filet.close()
    
    def reset(self):
        """Clear all stored trajectories and object tracking data."""
        self.center_points.clear()
        self.trajectories.clear()
        self.id_count = 0