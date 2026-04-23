import json, math, os
import numpy as np
from controller import Robot, Supervisor


def get_size(node):
    if node is None:
        return None
    size_node_field = node.getField("size")
    if size_node_field is not None:
        return size_node_field.getSFVec3f()
    
    bounding_field = node.getField("boundingObject")
    if bounding_field is None:
        return None

    bounding_node = bounding_field.getSFNode()
    if bounding_node is None:
        return None

    size_field = bounding_node.getField("size")
    if size_field is None:
        return None

    return size_field.getSFVec3f()



def build_environment_map(supervisor):
    
    environment_map = {"pick_objects" : {},
                       "navigation_goals" : {},
                       "place_zone" : {},
                       "obstacles": []}
    obstacles = ["pick_table_1", "pick_table_2","nav_goal", "obs1", "obs2", "obs3", "obs4", "obs5", "obs6","wall_inner_e","wall_inner_w","wall_n","wall_s", "wall_w", "wall_e"]
    objects = ["OBJECT_1","OBJECT_2"]
    navigation_goals = ["OBJECT_1","OBJECT_2"] 
    place_zone = ["nav_goal"]

    for def_name in obstacles:
        node = supervisor.getFromDef(def_name)
        if node:
            pos = node.getPosition()
            size = get_size(node)
            orientation = node.getOrientation()

            yaw = float(__import__("math").atan2(orientation[3], orientation[0]))
            environment_map["obstacles"].append({"def_name" : def_name, "position": pos,"size": size, "yaw_radians" : yaw})
            
    # for def_name in walls:
    #     node = lab7_supervisor.getFromDef(def_name)
    #     if node:
    #         pos = node.getPosition()
    #         size = get_size(node)
    #         environment_map["walls"].append({"def_name" : def_name, "position": pos,"size": size})
    
    for def_name in objects:
        node = supervisor.getFromDef(def_name)
        if node:
            pos = node.getPosition()
            size = get_size(node)
            pos[1] = pos[1] + 0.6
            orientation = node.getOrientation()

            yaw = float(__import__("math").atan2(orientation[3], orientation[0])) - np.pi/2
            environment_map["pick_objects"][def_name] = ({"position": pos, "size": size, "yaw_radians" : yaw})
            
    for def_name in navigation_goals:
        node = supervisor.getFromDef(def_name)
        if node:
            pos = node.getPosition()
            orientation = node.getOrientation()
            pos[1] = pos[1] + 0.6

            yaw = float(__import__("math").atan2(orientation[3], orientation[0])) - np.pi/2
            environment_map["navigation_goals"][def_name] = ({"position" : pos, "yaw_radians" : yaw})
    for def_name in place_zone:
        node = supervisor.getFromDef(def_name)
        if node:  
            pos = node.getPosition()
            orientation = node.getOrientation()
            pos[1] = pos[1] - 1.3
            yaw = float(__import__("math").atan2(orientation[3], orientation[0])) + np.pi/2
            environment_map["place_zone"][def_name] = ({"position": pos, "yaw_radians" : yaw})

    return environment_map
    # with open('../lab7_pr2/environment_map.json', 'w') as f:
    #     json.dump(environment_map, f, indent=4)
    # print("Map saved to environment_map.json")
    # with open("../lab7_pr2/environment_map.json", 'r') as file:
    #     # json.load() parses the file and returns a dict or list
    #     return json.load(file)


def navigate_to_goal(pr2, goal_x, goal_y, goal_yaw, env_map):
    raise NotImplementedError("TODO 8: Implement navigate_to_goal()")

def write_robot_pose(supervisor):
    robot_node = supervisor.getFromDef("PR2")
    if robot_node is None:
        print("Could not find PR2 node")
        return

    pos = robot_node.getPosition()
    orientation = robot_node.getOrientation()

    yaw = float(__import__("math").atan2(orientation[3], orientation[0]))

    pose_data = {
        "x": pos[0],
        "y": pos[1],
        "yaw": yaw
    }

    with open("../lab7_pr2/robot_pose.json", "w") as f:
        json.dump(pose_data, f, indent=4)
    
#Do not modify the main
def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    
    
    env = build_environment_map(supervisor)
    
    with open('../lab7_pr2/environment_map.json', 'w') as f:
        json.dump(env, f, indent=4)
        
    print("Map saved to environment_map.json")
    
    while supervisor.step(timestep) != -1:
        write_robot_pose(supervisor)
    
    
    # print(env)
if __name__ == "__main__":
    main()