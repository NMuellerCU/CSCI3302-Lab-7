import json, math, os
import numpy as np
from controller import Robot, Supervisor

def load_environment_map():
    lab7_supervisor = Supervisor()
    
    environment_map = {"pick_objects" : {}, "navigation_goals" : {}, "place_zone" : {}, "obstacles": []}
    obstacles = ["pick_table_1", "pick_table_2", "obs1", "obs2", "obs3", "obs4", "obs5", "obs6", "wall_inner_e", "wall_inner_w"]
    objects = ["OBJECT_1","OBJECT_2"]
    navigation_goals = ["OBJECT_1","OBJECT_2"] 
    place_zone = ["nav_goal"]
    print(lab7_supervisor.getTime())
    
    if lab7_supervisor.getTime() >= 0.0:
        for def_name in obstacles:
            node = lab7_supervisor.getFromDef(def_name)
            if node:
                pos = node.getPosition()
                environment_map["obstacles"].append({
                "def_name" : def_name, 
                "position": pos})
                
        
        for def_name in objects:
            node = lab7_supervisor.getFromDef(def_name)
            if node:
                
                environment_map["pick_objects"].update({def_name: None})
                
        for def_name in navigation_goals:
            node = lab7_supervisor.getFromDef(def_name)
            if node:
                pos = node.getPosition()
                yaws = node.getOrientation()
                
                environment_map["navigation_goals"].update({def_name: None})
                environment_map["navigation_goals"][def_name] = {"position": pos, "yaw_radians": yaws}
                
        node = lab7_supervisor.getFromDef("place_zone")
        if node:
            pos = node.getPosition()
            yaws = node.getOrientation()
            
            environment_map["place_zone"]["nav_goal"] = {"position": pos, "yaw_radians": yaws}

        with open('../lab7_pr2/environment_map.json', 'w') as f:
            json.dump(environment_map, f, indent=4)
        print("Map saved to environment_map.json")
        with open("../lab7_pr2/environment_map.json", 'r') as file:
            # json.load() parses the file and returns a dict or list
            return json.load(file)


def navigate_to_goal(pr2, goal_x, goal_y, goal_yaw, env_map):
    raise NotImplementedError("TODO 8: Implement navigate_to_goal()")


#Do not modify the main
def main():
    env = load_environment_map()
    # print(env)
if __name__ == "__main__":
    main()