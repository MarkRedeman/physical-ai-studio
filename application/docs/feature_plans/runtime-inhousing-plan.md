# Runtime Inhousing Plan


Since this will be a rather big refactor we will work on it in steps using a feature branch, only merging said feature branch at the end. Now onto the architecture. A new RuntimeSession is created. This session is responsible for handling lifecycle of the session. This session can load an environment, dataset, model and control.

## Lifecycle handling

A lot of components in these acquire real hardware. This means that it needs to be acquired properly and released once done.
Same goes for Processes that are started which need to be terminated when the RuntimeSession stops.

In order to do this we will use contextmanagers. The goal is to have the lifecycle be handled by the thread or process itself:

```python

class EnvironmentContext:
    follower: RobotWorker
    leader: RobotWorker | None
    cameras: dict[str, CameraWorker]

class EnvironmentWorker(BaseThreadWorker[EnvironmentContext]):
   def __init__(self, environment: EnvironmentWithRelations, stop_event: EventClass, daemon: bool = False):
        super().__init__(stop_event=stop_event, daemon=daemon)
        self.environment_config = environment
      
   async def lifecycle(self):
        # one robot for now until refactor of environment
        try:
            robot = self.environment_config.robots[0]

            #logic to get robot from config:
            follower = self._robot_worker(robot)
            leader = None
            if robot.tele_operator:
                leader = await self._robot_worker(robot.tele_operator.robot)

            cameras = { camera.name, await self._camera_worker(camera):  for camera in self.environment_config.cameras}

            yield EnvironmentContext(
                follower=follower,
                leader=leader,
            )
        except Exception as e:
            logger.exception(f"Unhandled exception in {self.name}")
            raise e
        finally:
            follower.disconnect()
            if leader:
                leader.disconnect()
            
    async def run_loop(self, context: EnvironmentContext):
        # do things with context.
      
```

The BaseProcessWorker or BaseThreadWorker will inherit from thread or mp.process and critically the run function will do this:
```Python

def run(self) -> None:
    async with self.lifecycle() as context:
        self.run_loop(context)
```



## Communication and Data Store

Some of these components live in different processes for stability. This means that cross process communication needs to be established. Two will be added:

1. A Store

A store where the robot state, actions and observations, task (current joint state, next joint state, camera views respectively and task) are kept. 
These are kept in SharedMemory so that all components can easily read and write to it.
The shape of the store depends on what was loaded first.
A dataset for example has a specific feature set (robot state and cameras), the actions are simply a mirror of the robot state.
A model has specific model inputs which decide the shape and an Environment has robots and cameras which decide the robot state, actions and observations.
The store initially only has state (array), action (array)  and observations (dict of arrays).
This distills down to state being a Sample.
The Sample contains the shared memory, the feature names (list of joints or just a name in case of observation (e.g. overhead, wrist)) and a timestamp of when it was set. 

2. Events over Zenoh. 

To be able to easily handle events components need to be able to listen for specific events over zenoh. The consumer of the session (e.g. a websocket connection from fastapi) needs to be able to adjust things in the subcomponents. 

## Workers

Most components run on their own frequency. A dataset can be recorded at 50hz, robot at 100hz and model at 30hz for example. It is then also possible to run inference using a model of 30hz but record at 50hz. Yes, there will be some stepping, but that is fine. 

### Robot

Every robot needs their own worker in a process. It connects to the robot and disconnects it after done. It runs on a specific hz and takes writes their features to Store state and reads the action from the Store and writes it to the robot using the timestamp to get the goal_time. Not every robot uses goal_time and that is fine for now. No complex interpolation

### Camera

CameraWorker is a thread worker that starts a shared_camera from physicalai library. It then writes to the Store's observation shared memory of that camera at the desired hz.

### Dataset

A dataset worker runs in a separate process. It listen for start, stop, discard recording events. Once recording it writes the store's values as a frame to the dataset. This is at a desired hz. 

### Control

In order to allow for very flexible control of the entire system we need a control system. 
1. TeleoperationControl reads from leader and writes to the store.action.
2. ModelControl takes a loaded model and reads the current action from the model giving the current store frame and writes it to the store.action.
3. MoveControl will gradually move the robot from the current position to the target position over a time by writing to the store.action.

However, it needs more flexibility than just writing to the storte.action. A future human-in-the-loop Control would have to deal with the model, leader and perhaps even set the leader position. To make the Control as powerful as possible it is ran in a seperate process or thread and it can get zenoh events. Allowing for seperate hz's for whatever purpose it needs. Only one control can be activate at the same time.

## RuntimeSession

RuntimeSession all ties this together. The RuntimeSession holds the current store. On load_environment it will do so. Detailed errors will be thrown when a load_dataset (or other components) is called with a config that does not match current feature spec of the store. All subworkers are added to the Context of the RuntimeSession so that it cleans up after itself.
