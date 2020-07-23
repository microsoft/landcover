# To-do list

- [ ] Move some of these To-do items to github issues so that we can reference them
- [ ] Show download processing status on the front-end (index.html)
  - [ ] When you press download there should be some sort of status indiator on the front-end that changes when the download is done / fails.
  - [ ] (Potentially) The front-end should not wait for results on the same HTTP request that a download was initiated on and should instead poll for results. 
- [ ] Session poll thread on the front-end (index.html)
  - [ ] Every 10ish seconds the front-end should poll an endpoint to ask what the status of its session is.
  - [ ] The status should be displayed somewhere on the page
  - [ ] If the session has died then some blocking indication should be given
- [ ] Create small debug page (e.g. `/sessions.html`) that shows a list of the active sessions
- [ ] Make the Noty notifactions have a theme that matches the rest of the app
- [ ] Dummy password authentication. (landing_page.html)
  - [ ] The landing page should have a hardcoded password prompt that you must pass in order to use the page. Passing this prompt should set a 1-day cookie that bypasses it. 
- [ ] Total rework of model saving and loading. (everywhere)
  - [ ] Currently the tool generates a custom link that a model can be restored at however this is brittle and unintuitive to users.
  - [ ] Rename ServerModelsAbstract to ModelSessionAbstract throughout.
    - [ ] Clean up (remove NAIP references) and re-document the interface
    - [ ] Add `save_state_to()` and `load_state_from()` methods to the interface. Now, "ServerModels" will be responsible for serializing their state to a directory.
  - [ ] Add a checkpoint model button to the front-end.
    - [ ] This should prompt for a checkpoint name.
    - [ ] This should save the model to disk.
    - [ ] This should save an entry in a checkpoint database.
  - [ ] The landing page should have an additional section that shows available checkpoints for each (dataset, model) pair. Additionally the landing page should give the option to start from an empty model. 
- [ ] Add ability for multiple named basemaps to be passed through the entries in `datasets.json`
  - [ ] The front end should display all basemaps for the selected dataset.
  - [ ] Add a hotkey to switch between the custom basemap layers
- [ ] Checkpoint handling on the landing page
  - [x] Add "valid_models" key to each dataset that is a list of acceptable models.
  - [ ] The expected flow is: "user selects a dataset" --> "valid list of models are displayed" --> "user selects a model" --> "current list of checkpoints are displayed" --> "user selects a checkpoint or 'new'" --> "start server button is enabled"
  - [ ] The "current list of checkpoints" should have icons that allow them to be renamed or deleted.
  - [ ] When a checkpoint is selected we should display useful information about it: how many points, how many sessions, etc.
- [ ] The commnication between `server.py` and `worker.py` needs to be re-worked.
  - Currently `server.py` will spawn an instance of `worker.py` for every "session" that is created through the front-end. Communication between the server and the worker are handled by `rpyc` RPC calls. In long running computations on the worker (e.g. running a model over a tile), the connection will time-out. Also, the RPC call seems to incur a significant overhead when passing large arrays (e.g. a 7000x7000x20 numpy array).
  - Celery with a Redis backend might be a good solution here.
  - [ ] Move this to an github issue
- [ ] Create actual `unittest` test cases for most server functionality
  - [ ] Use the existing master branch to get expected responses from various functions
  - [ ] Convert the existing cases to use `unittest`
- [ ] The front-end does not need to call `get_input()` or display the image/model results in the top-right corner
- [ ] Rebase/merge the feature/cycle branch
- [ ] Implement a small tutorial screen / page that explains how to use the tool


## Documentation

- [ ] Document the existing API exposed by `server.py`, create `API.md`
- [ ] Create simple instructions for how to use the tool to include in `README.md`
- [ ] Document the DataLoaderAbstract interface
