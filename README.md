# Deep Q Learning Example

Simple project to test a deep q neural network quickly.

## Results

Untrained network<br/>
![Untrained](https://github.com/adza45/qtests/blob/master/media/Untrained.gif)

Trained network (after approximately 13000000 game loops)<br/>
![Trained](https://github.com/adza45/qtests/blob/master/media/Trained.gif)

## How to run

Clone this repo

To run with the game screen showing:<br/>
python3 game.py --load False --display True

To run without the game screen showing: (This will train significantly faster)<br/>
python3 game.py --load False --display False

## Reference

Neural net code based from:<br/>
https://github.com/asrivat1/DeepLearningVideoGames
