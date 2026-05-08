
# ML Project February 2026: Knights Archers Zombies

This repository contains the code to setup the final evaluation of the course "[Machine Learning: Project](https://onderwijsaanbod.kuleuven.be/syllabi/e/H0T25AE.htm)" (KU Leuven, Faculty of Engineering, Department of Computer Science, [DTAI Section](https://dtai.cs.kuleuven.be)).


## Summary

- Install all dependencies in `requirements.txt`
- Train two agents using an RL library. See `example_training_rllib.py` for an example of training simple agents using Ray RLLib.
- The KAZ environment is created by using the corresponding function in the `utils.py` file. 
- During your project, you can test your trained agent via the `evaluation.py` file. To know about the options, run `python3 evaluation.py -h`.
- For the final submission, you need to implement the interfaces in the file `submission.py` for the two archers (Task 3) environment. 
- Submit your own code on the departmental computers for the tournament (see below). See the file `submission_example_rllib.py` for an example.


## Use on departmental computers

The departmental computers will be used to run a tournament and submit your implementation (see detailed instructions below). You can also use these computers to train your agents. A tutorial to connect remotely via SSH can be found [here](ssh.md) and additional info is available on [the departmental web pages](https://system.cs.kuleuven.be/cs/system/wegwijs/computerklas/index-E.shtml).

You will see a personal directory in:

```
/cw/lvs/NoCsBack/vakken/H0T25A/ml-project
```

There is an upper limit of 50MB on the disk space that you can use. Remote (ssh) users are also limited to 2GB of RAM.

PyGame, PettingZoo and other packages that you can use are pre-installed in a virtual environment, which can be activated using:

```
source /cw/lvs/NoCsBack/vakken/H0T25A/ml-project/venv/bin/activate
```

Since this virtual environment will be used to run the tournament, you should avoid language features that are not compatible with the installed Python version (3.12) or use packages that are not installed. All of PettingZoo's [butterfly](https://pettingzoo.farama.org/content/basic_usage/) dependencies are currently installed, as well as `torch==2.10.0` and `tensorflow==2.20.0`.


## Local installation

- It is recommended to use a newly-created virtual environment to avoid dependency conflicts.


- Install Pettingzoo with the additional requirements for the Butterfly environments

    ```
    pip install 'pettingzoo[butterfly]'
    ```

- Install SuperSuit, which will help managing your environments:

    ```
    pip install supersuit
    ```

- Your agents will be dependent on some RL library. Here we provide an example for installing Ray RLlib:

    ```
    pip install 'ray[rllib]'
    ```

- All dependencies are also listed in the `requirements.txt` file (`pip install -r requirements.txt`). Or in the `pyproject.toml` file if you want to use `uv`.

You can (visually) test your installation by running:

```
python3 evaluation.py -l random_agent.py -s --distortion=5
```


## Zombie detection

An unseen test set of observations will be given to your agent. Your agent should return the bounding boxes of all the zombies it has found. Based on intersection over union, the average precision is computed. You can test the setup with an example data file in the `observation_data` folder and running:

```
python3 evaluation.py -l random_agent.py --zombies
```


## Tournament

The tournament will be played with agents that are available on the departmental computers. This will allow you to try your agents in the identical environment that is used by the tournament script. For this to work, you have to adhere to the following setup:

- Your agents implement the interface in the `submission.py` 
- The tournament code will scrape the entire directory provided for you on the departmental computers for the `submission.py` file. If multiple matching files are found, a random one will be used.
- Your agents should be ready to play in a few seconds, thus use pre-trained policies. An agent that is not responding after 10 seconds will forfeit the game.
- There is no timeout on the actions. The required speed is defined by the zombies that move down. Check your code on the departmental computers to get an idea of how fast your code runs. Or implement a timeout yourself to guarantee fast enough actions.

### Paths

Make sure you **do not use relative paths** in your implementation to load your trained model, as this will fail when running your agent from a different directory. Best practice is to retrieve the absolute path to the module directory:

```python
package_directory = os.path.dirname(os.path.abspath(__file__))
```

Afterwards, you can load your resources based on this `package_directory`:

```python
model_file = os.path.join(package_directory, 'models', 'mymodel.pckl')
```

## Submission using the Departmental Computers

To submit your agent, a copy of your code and agent needs to be available on the departmental computers in a directory assigned to you. Also the code to train your agent should be included.

The departmental computers have the `requirements.txt` packages already installed such that you can verify that your agent works. During the semester the tournament script will be run to play games between the (preliminary) agents that are already available. A tentative ranking will be shared.


## FAQ

### Where do I ask questions?

On the Toledo Discussion board or in the Q&A sessions.

### I found a mistake in the template or the examples in this repository

Submit a Github pull request.

### Installation cannot find Tensorflow

Tensorflow is only compatible with Python 3.9‒3.12.

On macOS you can use an older Python version by running these commands before the install script:

```
brew install python@3.12  # if using homebrew
virtualenv -p /usr/local/opt/python@3.12/bin/python3 venv
. ./venv/bin/activate
```

### Getting error during the installation of the requirements

Consider downgrading your python version to 3.12 or lower as some packages might not yet support the latest python version.

