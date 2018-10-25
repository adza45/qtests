import numpy as np
import random
import cv2
from objects import Unit, Player, Objective, Mine
from time import sleep
from agent import Agent
import optparse
import sys
import time
import tensorflow as tf
import pickle
from collections import deque
import json

##########################################################################

'''Game Parameters'''
MAP_SIZE = (65, 65)

##########################################################################

''' Units Setup '''
def init():
	units = []
	player = Player(MAP_SIZE)
	units.append(player)
	player.set_location((random.randint(0,12) * 5, random.randint(0,12) * 5))
	player.inital_location = player.location

	objective = Objective()
	units.append(objective)
	objective.set_location(player.location)

	while (player.location == objective.location):
		objective.set_location((random.randint(0,12) * 5, random.randint(0,12)  * 5))

	mine_count = 10
	mines = []
	for x in range(mine_count):
		temp_location = player.location
		while (temp_location == player.location) or (temp_location == objective.location):
			temp_location = (random.randint(0,12) * 5, random.randint(0,12) * 5)
		mines.append(Mine())
		units.append(mines[x])
		mines[x].set_location(temp_location)

	return units

##########################################################################
def draw(units, options):
	game_data = np.zeros((MAP_SIZE[0], MAP_SIZE[1], 3), np.uint8)
	for unit in units:
		cv2.rectangle(game_data, (unit.location[0], unit.location[1]), (unit.location[0] + 4, unit.location[1] + 4), unit.color, -1)

	image = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)

	if options.display == "True":
		resized = cv2.resize(image, dsize=None, fx=4, fy=4)
		cv2.imshow("Stacked Image", resized)
		cv2.waitKey(1)

	return image

##########################################################################

def move(choice, player):
	if choice == "w":
		player.move_up()
	elif choice == "d":
		player.move_right()
	elif choice == "s":
		player.move_down()
	elif choice == "a":
		player.move_left()

##########################################################################

def update(units, total_score):
	player = units[0]
	agent.r_t = -0.001
	for unit in units:
		if isinstance(unit, Mine) and unit.location == player.location:
			player.score -= 1
			player.set_location(player.inital_location)
			units.remove(unit)
			agent.r_t = -1
			total_score -= 1
		elif isinstance(unit, Objective) and unit.location == player.location:
			player.score += 1
			player.restart = True
			agent.r_t = 1
			total_score += 1

	return total_score

units = init()
agent = Agent()

def main(options, units, agent):
	game_step = 0

	run = True

	image = draw(units, options)
	agent.x_t = image
	print("SHAPE: {}".format(agent.x_t.shape))
	agent.r_t = 0
	agent.s_t = np.stack((agent.x_t, agent.x_t, agent.x_t, agent.x_t), axis=2)

	if options.load == "True":
		if agent.checkpoint and agent.checkpoint.model_checkpoint_path:
			agent.saver.restore(agent.sess, agent.checkpoint.model_checkpoint_path)
			print("Successfully loaded:", agent.checkpoint.model_checkpoint_path)
		else:
			print("Could not find old network weights")

		with open('saved_networks/D.pkl', 'rb') as input:
			agent.D = pickle.load(input)

		with open('saved_networks/tracker.json') as f:
			tracker = json.load(f)

			agent.t = tracker["t"]
			agent.epsilon = tracker["epsilon"]

			agent.EXPLORE = tracker["explore"] - (tracker["t"] - agent.OBSERVE)

	else:
		print("starting new training session. This will overwright saved weights.")

	'''Main Game Loop'''
	total_score = 0
	while run:

		total_score = update(units, total_score)

		image = draw(units, options)

		agent.terminal = False
		if units[0].restart == True:
			agent.terminal = True
			units = init()
			image = draw(units, options)
			agent.x_t = image
			agent.s_t1 = np.stack((agent.x_t, agent.x_t, agent.x_t, agent.x_t), axis=2)
		else:
			agent.x_t1 = image
			agent.x_t1 = np.reshape(agent.x_t1, (MAP_SIZE[0], MAP_SIZE[1], 1))
			agent.s_t1 = np.append(agent.x_t1, agent.s_t[:, :, :3], axis=2)

		agent.last_time = time.time()

		agent.D.append((agent.s_t, agent.a_t, agent.r_t, agent.s_t1, agent.terminal))
		if len(agent.D) > agent.REPLAY_MEMORY:
			agent.D.popleft()

		if agent.t > agent.OBSERVE:
			minibatch = random.sample(agent.D, agent.BATCH)

			s_j_batch = [d[0] for d in minibatch]
			a_batch = [d[1] for d in minibatch]
			r_batch = [d[2] for d in minibatch]
			s_j1_batch = [d[3] for d in minibatch]
			y_batch = []
			readout_j1_batch = agent.readout.eval(feed_dict = {agent.s : s_j1_batch})
			for i in range(0, len(minibatch)):
				agent.terminal = minibatch[i][4]

				if agent.terminal:
					y_batch.append(np.clip(r_batch[i], -1, 1))
				else:
					error = np.clip(r_batch[i] + agent.GAMMA * np.max(readout_j1_batch[i]), -1, 1)

					y_batch.append(error)

			agent.train_step.run(feed_dict = {
				agent.y : y_batch,
				agent.a : a_batch,
				agent.s : s_j_batch}
			)

		agent.s_t = agent.s_t1
		agent.t = agent.t + 1

		if agent.t % 10000 == 0:
			agent.saver.save(agent.sess, 'saved_networks/' + agent.GAME + '-dqn', global_step = agent.t)

			with open('saved_networks/D.pkl', 'wb') as output:
				pickle.dump(agent.D, output, pickle.HIGHEST_PROTOCOL)

			with open('saved_networks/tracker.json', 'w') as outfile:
				save_network = {
					"t": agent.t,
					"epsilon": agent.epsilon,
					"explore": agent.EXPLORE
					}
				json.dump(save_network, outfile)

		state = ""
		if agent.t <= agent.OBSERVE:
			state = "observe"
		elif agent.t > agent.OBSERVE and agent.t <= agent.OBSERVE + agent.EXPLORE:
			state = "explore"
		else:
			state = "train"
		print("TIMESTEP", agent.t, "/ STATE", state, \
			"/ EPSILON", agent.epsilon, "/ ACTION", agent.action_index, "/ REWARD", agent.r_t, \
			"/ Q_MAX %e" % np.max(agent.readout_t), "/ SCORE", total_score)

		# choose an action epsilon greedily
		agent.readout_t = agent.readout.eval(feed_dict={agent.s : [agent.s_t]})[0]
		agent.a_t = np.zeros([agent.ACTIONS])
		agent.action_index = 0

		if random.random() <= agent.epsilon:
			#print("----------Random Action----------")
			agent.action_index = random.randrange(agent.ACTIONS)
			agent.a_t[random.randrange(agent.ACTIONS)] = 1
		else:
			agent.action_index = np.argmax(agent.readout_t)
			agent.a_t[agent.action_index] = 1

		if agent.epsilon > agent.FINAL_EPSILON and agent.t > agent.OBSERVE:
			agent.epsilon -= (agent.INITIAL_EPSILON - agent.FINAL_EPSILON) / agent.EXPLORE

		# choice = input('Move: ')
		# move(choice, units[0])

		if agent.a_t[0] == 1:
			move("d", units[0])
		elif agent.a_t[1] == 1:
			move("s", units[0])
		elif agent.a_t[2] == 1:
			move("a", units[0])
		elif agent.a_t[3] == 1:
			move("w", units[0])
		elif agent.a_t[4] == 1:
			pass

		game_step = game_step + 1

def validate_options(options):
	if options.load != "True" and options.load != "False":
		print("Arguement 'load' must be 'True' or 'False'")
		sys.exit(0)

	if options.display != "True" and options.display != "False":
		print("Arguement 'display' must be 'True' or 'False'")
		sys.exit(0)

if __name__ == '__main__':

	parser = optparse.OptionParser()

	parser.add_option('-l', '--load', action="store", dest="load", help="Load previous weights (True/False)", default="True")
	parser.add_option('-d', '--display', action="store", dest="display", help="Display game screen", default="True")

	options, args = parser.parse_args()

	validate_options(options)

	main(options, units, agent)
