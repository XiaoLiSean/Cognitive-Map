from lib.robot_env import Agent_Sim
exec_path = "/Users/Jake/MastersFirstYear/PROGRESS_LAB/ai2thor/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-Thor"

robot = Agent_Sim(exec_path=exec_path)
robot.reset_scene(scene_type="Dynamics", scene_num=4, ToggleMapView=True, Show_doorway=False, shore_toggle_map=False)
robot.show_map(show_nodes=False, show_edges=False)
