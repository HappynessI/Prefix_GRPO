import os
import json
import threading
import traceback
from .environment import SingleAlfredTWEnv
from .utils import load_config, process_ob


def _positive_int_from_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _nonnegative_int_from_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


class ALFWorld_Wrapper:
    def __init__(self, **kwargs):
        # load data_path
        self.data_path = kwargs.get("data_path", None)
        if self.data_path is None:
            raise Exception("missing parameter data_path")
        os.environ["ALFWORLD_DATA"] = self.data_path

        # load config for alfworld benchmark
        self.config_path = kwargs.get("config_path", None)
        if self.config_path is None:
            raise Exception("missing parameter config_path")
        self.config = load_config(self.config_path)

        self._max_id = 0
        self.ls = []
        self.env = {}  # dict[id, env_item]
        self.env_init = {}  # dict[id, env_item]
        self.info = {}  # dict[id, env_info]
        self.games = []  # list[game_file]
        self._created_total = 0
        self._closed_total = 0
        self._failed_total = 0
        self._active_high_watermark = 0
        self._active_leases = set()
        self._max_active_envs = _nonnegative_int_from_env("ALFWORLD_MAX_ACTIVE_ENVS_PER_SERVER", 1)
        self._lock = threading.Lock()
        self._op_semaphore = threading.BoundedSemaphore(
            _positive_int_from_env("ALFWORLD_ENV_OP_CONCURRENCY", 1)
        )
        self._active_semaphore = (
            threading.BoundedSemaphore(self._max_active_envs)
            if self._max_active_envs > 0
            else None
        )
        
        train_games_root = os.path.join(
            os.environ["ALFWORLD_DATA"], "json_2.1.1", "train"
        )
        test_games_root = os.path.join(
            os.environ["ALFWORLD_DATA"], "json_2.1.1", "valid_train"
        )

        train_mapping_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "configs",
            "mappings_train.json",
        )
        test_mapping_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "configs",
            "mappings_test.json",
        )

        with open(train_mapping_file, "r") as f:
            mappings = json.load(f)
            for mapping in mappings:
                self.games.append(
                    os.path.join(
                        train_games_root,
                        mapping["task_type"],
                        mapping["task_id"],
                        "game.tw-pddl",
                    )
                )

        with open(test_mapping_file, "r") as f:
            mappings = json.load(f)
            for mapping in mappings:
                self.games.append(
                    os.path.join(
                        test_games_root,
                        mapping["task_type"],
                        mapping["task_id"],
                        "game.tw-pddl",
                    )
                )

    def create(self):
        acquired_active_slot = False
        try:
            if self._active_semaphore is not None:
                acquired_active_slot = self._active_semaphore.acquire(blocking=False)
                if not acquired_active_slot:
                    with self._lock:
                        active = len(self._active_leases)
                    return {
                        "error": (
                            "ALFWorld server active environment capacity reached: "
                            f"{active}/{self._max_active_envs}"
                        ),
                        "error_code": "active_capacity",
                        "retryable": True,
                        "active_env_capacity": self._max_active_envs,
                        "active_env_leases": active,
                    }

            # TODO extend to other kinds of environments
            with self._op_semaphore:
                with self._lock:
                    idx = self._max_id
                    self._max_id += 1
                    if acquired_active_slot:
                        self._active_leases.add(idx)
                self.env[idx] = SingleAlfredTWEnv(self.config)
                self.info[idx] = {"done": False, "reward": 0, "deleted": False}
                self.ls.append(idx)
                with self._lock:
                    self._created_total += 1
                    active = len(self.env)
                    self._active_high_watermark = max(self._active_high_watermark, active)
                if idx < 5 or idx % 100 == 0:
                    print(
                        f"-------Env {idx} created, active={active}, "
                        f"created_total={self._created_total}, closed_total={self._closed_total}--------",
                        flush=True,
                    )
            payload = {"id": idx}
        except Exception as e:
            if acquired_active_slot:
                self._release_active_slot(locals().get("idx", None))
            self._log_exception("create", None, e)
            payload = {"error": f"{e}"}
        return payload
    
    def __del__(self):
        for idx in list(self.ls):
            env_init = self.env_init.pop(idx, None)
            if env_init is not None:
                env_init.close()
                print(f"-------Env {idx} closed--------", flush=True)

    def step(self, idx: int, action: str):
        try:
            with self._op_semaphore:
                self._check_id(idx)
                ob, _, done, info = self.env_init[idx].step([action])
                ob, reward, done = process_ob(ob[0]), float(info["won"][0]), done[0]
                available_actions = info.get("admissible_commands", [[]])[0]
                payload = {
                    "observation": ob,
                    "reward": reward,
                    "available_actions": available_actions,
                    "done": done,
                }
                self.info[idx].update(payload)
        except Exception as e:
            self._log_exception("step", idx, e)
            payload = {"error": f"{e}"}
        return payload

    def reset(self, idx: int, game: int, world_type: str):
        if world_type not in ["Text", "Embody", "Hybrid"]:
            return {"error": 'world_type must be one of "Text", "Embody" and "Hybrid"'}
        try:
            with self._op_semaphore:
                self._check_id(idx, True)
                self.env[idx].game_files = [self.games[game]]
                self.env[idx].num_games = 1
                self.env_init[idx] = self.env[idx].init_env(batch_size=1)
                ob, info = self.env_init[idx].reset()
                ob = "\n".join(ob[0].split("\n\n")[1:])
                available_actions = info.get("admissible_commands", [[]])[0]
                payload = {
                    "id": idx,
                    "observation": ob,
                    "available_actions": available_actions,
                    "task_type": "/".join(info["extra.gamefile"][0].split("/")[-3:-1]),
                }
                self.info[idx] = {
                    "world_type": world_type,
                    "game": game,
                    "observation": ob,
                    "available_actions": available_actions,
                    "done": False,
                    "reward": 0,
                    "deleted": False,
                }
        except Exception as e:
            self._log_exception("reset", idx, e)
            payload = {"error": str(e)}
        return payload

    def get_observation(self, idx: int):
        try:
            self._check_id(idx)
            return self.info[idx]["observation"]
        except Exception as e:
            return {"error": str(e)}

    def get_available_actions(self, idx: int):
        try:
            self._check_id(idx)
            return self.info[idx]["available_actions"]
        except Exception as e:
            return {"error": str(e)}

    def get_detailed_info(self, idx: int):
        try:
            self._check_id(idx)
            return self.info[idx]
        except Exception as e:
            return {"error": str(e)}

    def close(self, idx: int):
        try:
            with self._op_semaphore:
                # Finished episodes must still be closable; otherwise done=True
                # trajectories leak in env/env_init/info after successful rollout.
                self._check_id(idx, is_reset=True)
                env_init = self.env_init.pop(idx, None)
                if env_init is not None:
                    env_init.close()
                self.env.pop(idx, None)
                self.info.pop(idx, None)
                try:
                    self.ls.remove(idx)
                except ValueError:
                    pass
                self._release_active_slot(idx)
                with self._lock:
                    self._closed_total += 1
                    active = len(self.env)
                if idx < 5 or idx % 100 == 0:
                    print(
                        f"-------Env {idx} closed, active={active}, "
                        f"created_total={self._created_total}, closed_total={self._closed_total}--------",
                        flush=True,
                    )
            return {"id": idx, "closed": True, "active": active}
        except Exception as e:
            self._log_exception("close", idx, e)
            return {"error": str(e)}

    def stats(self):
        try:
            import textworld.gym
            textworld_registry_size = len(textworld.gym.registry)
        except Exception:
            textworld_registry_size = None

        with self._lock:
            active_ids = list(self.ls)
            return {
                "max_id": self._max_id,
                "created_total": self._created_total,
                "closed_total": self._closed_total,
                "failed_total": self._failed_total,
                "active_high_watermark": self._active_high_watermark,
                "max_active_envs": self._max_active_envs,
                "active_env_leases": len(self._active_leases),
                "active_env": len(self.env),
                "active_env_init": len(self.env_init),
                "active_info": len(self.info),
                "active_id_count": len(active_ids),
                "active_ids_head": active_ids[:20],
                "active_ids_tail": active_ids[-20:],
                "textworld_registry_size": textworld_registry_size,
            }

    def _log_exception(self, op: str, idx, exc: Exception):
        with self._lock:
            self._failed_total += 1
        print(f"ALFWorld env {op} failed id={idx}: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)

    def _release_active_slot(self, idx):
        if self._active_semaphore is None:
            return
        should_release = False
        with self._lock:
            if idx in self._active_leases:
                self._active_leases.remove(idx)
                should_release = True
        if should_release:
            self._active_semaphore.release()

    def _check_id(self, idx: int, is_reset: bool = False):
        if idx not in self.info:
            raise NameError(f"The id {idx} is not valid.")
        if self.info[idx]["deleted"]:
            raise NameError(f"The task with environment {idx} has been deleted.")
        if not is_reset and self.info[idx]["done"]:
            print("is reset", is_reset)
            print("done", self.info[idx]["done"])
            raise NameError(f"The task with environment {idx} has finished.")


os.environ["ALFWORLD_DATA"] = os.path.abspath(
    os.path.expanduser(os.environ.get("ALFWORLD_DATA") or os.environ.get("ALFWORLD_DATA_DIR") or "~/.cache/alfworld")
)
server = ALFWorld_Wrapper(
    data_path=os.environ["ALFWORLD_DATA"],
    config_path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "configs", "base_config.yaml"
    ),
)
