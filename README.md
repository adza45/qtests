# Deep Q Learning Example

Simple Deep Q project to test a deep q neural network quickly.

## Results

Completely random actions
![Untrained](https://github.com/adza45/qtests/blob/master/media/Untrained.gif)

Trained after approximately 13000000 game loops (t)
![Trained](https://github.com/adza45/qtests/blob/master/media/Trained.gif)

## How to run

Clone this repo

To run with the game screen showing:
python3 game.py --load False --display True

To run without the game screen showing: (This will train significantly faster)
python3 game.py --load False --display False

## Reference

Neural net code based from:
https://github.com/asrivat1/DeepLearningVideoGames
