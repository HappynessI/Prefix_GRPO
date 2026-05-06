from scienceworld import ScienceWorldEnv
import uuid
import threading

class SciWorldEnv:
    def __init__(self):
        self._max_id = 0
        self.env = {}
        self.info = {}
        self.games = []
        self.ls = []
        self._lock = threading.Lock()
        self._envlock = threading.Lock()
        exceptions = {"5-1", "5-2", "9-1", "9-2", "9-3", "10-1", "10-2"}
        init_env = ScienceWorldEnv()
        for key, value in init_env.tasks.items():
            if key not in exceptions:
                self.games += [
                    {"taskName": value, "variationIdx": i}
                    for i in range(init_env.getMaxVariations(value))
                ]
        init_env.close()
        del init_env

    def create(self):
        try:
            with self._lock:
                idx = self._max_id
                self._max_id += 1
            env = ScienceWorldEnv()
            with self._envlock:
                self.env[idx] = env
                self.info[idx] = {"deleted": False, "done": False}

            self.ls.append(idx)
            print(f"-------Env {idx} created--------")
            return {"id": idx}
        except Exception as e:
            return {"error": str(e)}

    def step(self, idx: int, action: str):
        try:
            self._check_id(idx)
            ob, reward, done, info = self.env[idx].step(action)
            payload = {
                "observation": ob,
                "reward": reward,
                "score": info["score"],
                "done": done,
            }
            self.info[idx].update(payload)
            return payload
        except Exception as e:
            return {"error": str(e)}


    def step_visual(self, idx: int, action: str):
        # Only used in visualization mode
        try:
            self._check_id(idx)
            processed_action = action
            if processed_action.endswith("</s>"):
                processed_action = processed_action[:-4]
            
            if "Action:" in processed_action:
                action_parts = processed_action.split("Action:")
                if len(action_parts) > 1:
                    processed_action = action_parts[1].strip()
                else:
                    processed_action = action_parts[0].strip()
            # print(f"-------Env {idx} step with action: {processed_action}--------")
            ob, reward, done, info = self.env[idx].step(processed_action)
            
            try:
                object_tree = self.env[idx].getObjectTree()
            except:
                object_tree = None
                
            try:
                inventory = self.env[idx].inventory()
            except:
                inventory = ""
                
            payload = {
                "observation": ob,
                "reward": reward,
                "score": info["score"],
                "done": done,
                "info": info,
                "object_tree": object_tree,
                "inventory": inventory,
                "moves": info.get("moves", 0)
            }
            self.info[idx].update(payload)
            return payload
        except Exception as e:
            return {"error": str(e)}

    def reset(self, idx: int, data_idx: int):
        try:
            self._check_id(idx, True)
            self.env[idx].load(
                self.games[data_idx]["taskName"], self.games[data_idx]["variationIdx"]
            )

            task_description = self.env[idx].getTaskDescription()
            ob, reward, done, info = self.env[idx].step("look around")

            payload = {
                "task_name": self.games[data_idx]["taskName"],
                "var_num": self.games[data_idx]["variationIdx"],
                "task_description": task_description,
                "observation": ob,
                "reward": reward,
                "score": info["score"],
                "deleted": False,
                "done": done,
            }
            self.info[idx].update(payload)
            return payload
        except Exception as e:
            return {"error": str(e)}

    def get_observation(self, idx: int):
        try:
            self._check_id(idx)
            return self.info[idx]["observation"]
        except Exception as e:
            return {"error": str(e)}

    def get_action_hint(self, idx: int):
        try:
            self._check_id(idx)
            return {
                "possible_actions": self.env[idx].getPossibleActions(),
                "possible_objects": self.env[idx].getPossibleObjects(),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_goals(self, idx: int):
        try:
            self._check_id(idx)
            return {"goals": self.env[idx].getGoalProgressStr()}
        except Exception as e:
            return {"error": str(e)}

    def get_detailed_info(self, idx: int):
        try:
            self._check_id(idx)
            return self.info[idx]
        except Exception as e:
            return {"error": str(e)}

    def _check_id(self, idx: int, is_reset: bool = False):
        if idx not in self.info:
            raise ValueError(f"The id {idx} is not valid.")
        if self.info[idx]["deleted"]:
            raise ValueError(f"The task with environment {idx} has been deleted.")
        if not is_reset and self.info[idx]["done"]:
            raise ValueError(f"The task with environment {idx} has finished.")

    def close(self,idx):
        if self.info[idx]["deleted"]:
            raise ValueError(f"The task with environment {idx} has been deleted.")
        self.env[idx].close()
        self.info[idx]["deleted"]=True
        self.ls.remove(idx)
        print(f"-------Env {idx} closed--------")
        return True
    # Below ONLY used in visualization mode
    def get_task_description(self, idx: int):
        try:
            self._check_id(idx)
            task_desc = self.env[idx].get_task_description()
            return {"task_description": task_desc}
        except Exception as e:
            return {"error": str(e)}
        
    def get_object_tree(self, idx: int):
        try:
            self._check_id(idx)
            object_tree = self.env[idx].getObjectTree()
            return {"object_tree": object_tree}
        except Exception as e:
            return {"error": str(e)}
        
    def get_current_state(self, idx: int):
        try:
            self._check_id(idx)
            state = {
                "observation": self.env[idx].look(),
                "inventory": self.env[idx].inventory(),
                "task_description": self.env[idx].get_task_description(),
                "goal_progress": self.env[idx].get_goal_progress(),
                "possible_actions": self.env[idx].get_possible_actions()[:10], 
                "possible_objects": self.env[idx].get_possible_objects()[:10], 
                "current_moves": self.env[idx].get_num_moves(),
                "environment_info": self.info[idx]
            }
            return state
        except Exception as e:
            return {"error": str(e)}

server = SciWorldEnv()
