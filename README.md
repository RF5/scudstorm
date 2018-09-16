# Scudstorm
Scudstorm is an agent parameterized by a deep neural network - trained via a genetic algorithm in Tensorflow eager - to play Entelect Software's 2018 Tower Defense game. 

As seen at the Deep Learning Indaba 2018 poster session. For reference, [here is a pdf version of the poster](http://goo.gl/MGZvYJ)

The aim of this project was 
- To see if I could make a genetic algorithm train large agents with autoregressive policies
- To see how efficient I could make an environment training pipeline between a Java game and python.

## File hierarchy description
- `/logs/` contains the tensorboard log files for each run.
- `/runs/` contains the directory where the python environment manager runs the games in parallel. Do not put things in this folder, it is automatically populated and managed from the python env scripts.
- `/refbot/` contains some initialization for the reference bot which plays against each agent.
- `/saves/` is where the keras `Model` saves are saved to.
- `/common/` contains various utilities, initialization and other scripts.
- `/deploy/` contains code for running on Entelect's server, optimized for speed of inference.

### Running training
To run Scudstorm, simply go (with one of the options indicated):
`python manager.py --mode [train, resume, test, rank]`
- `train` mode starts the training process from scratch, randomly initializing all the agents.
- `resume` resumes training from the agent saves in the `/saves/checkpoints/` folder.
- `test` runs various tests to check that your environment works correctly and is ready for training/playing. This is essentially "test your local setup"
- `rank` plays several games in a round-robin style to rank all the saves of agents in the `/saves/` folder. It outputs each agents win percentage and ELO score after it is done.

### Combining arbitrary tensorflow functions into the keras model (eager or not)
The keras `Model` does not like it if you try and do anything non-standard within the model (e.g. like doing a one-hot, sampling, even just a matmul tf op) between keras Layers in the model. To get around this, one must create keras Layers which wrap base tf/keras ops and then just use them as you would any other layers. So that is why we have this `custom_layers.py` which contains wrappers for a one-hot encoder and a categorical sampling from logits:
```python
class SampleCategoricalLayer(tf.keras.Layers.Layer):
   def call(self, x):
       dist = tf.distributions.Categorical(logits=x)
       return dist.sample()
class OneHotLayer(tf.keras.Layers.Layer):
   def call(self, x):
       meme = tf.keras.backend.one_hot(x, num_classes=self.num_classes)
       return meme
```

## Dependencies
The code should work on Windows 10, and newer versions of linux. Please run `requirements.txt` to get all necessary dependencies. Any recent version of Java is also required to simulate the Entelect Tower Defense game.

## Acknowledgements
1. Uber AI's paper on deep neuroevolution was an enormous inspiration and the core genetic algorithm I use is very similar to the one they use in their paper.
2. OpenAI baselines served as quite a large inspiration for parts of the environment pipeline, namely the `subproc_env_manager.py` code.
3. Parts of Inception-v4-resnet's network architecture was used for parts of Scudstorm's network. 
4. Entelect Software's 2018 Tower defense game is used. The .jar in this folder is their game/code. The `StarterBotPrime.py` is also a modified version of their python starter bot.

### Using this work
Feel free to use and modify this code really however you see fit, but please do acknowledge me as the original source.