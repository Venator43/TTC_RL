import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 3

HM_EPISODES = 5
MOVE_PENALTY = 1
LOSE_PENALTY = 350
WON_REWARD = 350
DRAW_REWARD = 175
epsilon = 0
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 0  # how often to play through env visually.

start_q_table ='qtable/qtable-1576227243-ttc-500000-new.pickle' # None or Filename

LEARNING_RATE = 0.1
DISCOUNT = 0.99

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}

def action(choice):
    '''
    Gives us 4 total movement options. (0,1,2,3)
    '''
    if choice == 0:
        return(0,0)
    elif choice == 1:
        return(0, 1)
    elif choice == 2:
        return(0, 2)
    elif choice == 3:
        return(1, 0)
    elif choice == 4:
        return(1, 1)
    elif choice == 5:
        return(1, 2)
    elif choice == 6:
        return(2, 0)
    elif choice == 7:
        return(2, 1)
    elif choice == 8:
        return(2, 2)


with open(start_q_table, "rb") as f:
    q_table = pickle.load(f)

episode_rewards = []

for episode in range(HM_EPISODES):
    board = [0,0,0,0,0,0,0,0,0]
    won_o = False
    lose_o = False
    episode_reward = 0
    turn = 0
    for i in range(9):
        turn += 1
        obs = tuple(board)
        if np.random.random() > epsilon and turn >1:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 9)

        b = 0
        for c in board:
            if c == 0:
                b += 1

       	if  b !=0:
            while board[action] != 0:
                action = np.random.randint(0, 9)
            board[action] = 1
        # Take the action!
        if board[0] == 1 and board[1] == 1 and board[2] == 1:
            won_o = True
        elif board[3] == 1 and board[4] == 1 and board[5] == 1:
            won_o = True
        elif board[6] == 1 and board[7] == 1 and board[8] == 1:
            won_o = True
        #horz
        elif board[1] == 1 and board[4] == 1 and board[7] == 1:
            won_o = True
        elif board[0] == 1 and board[3] == 1 and board[6] == 1:
            won_o = True
        elif board[2] == 1 and board[5] == 1 and board[8] == 1:
            won_o = True
        #dgonal
        elif board[0] == 1 and board[4] == 1 and board[8] == 1:
            won_o = True
        elif board[2] == 1 and board[4] == 1 and board[6] == 1:
            won_o = True
        
        n = 0
        for h in board:
            n+=1
            if h == 1:
                print("|O|",end="")
            elif h == -1:
               print("|X|",end="")
            elif h == 0:
               print("| |",end="")
            if n == 3:
                n=0
                print(" ")
             
        b = 0
        for c in board:
            if c == 0:
                b += 1

       	if  b !=0:
       	    action2 = (int(input("Move :")) - 1)
            while board[action2] != 0:
                action2 = (int(input("Move :")) - 1)
            board[action2] = -1
        #vert
        if board[0] == -1 and board[1] == -1 and board[2] == -1:
            lose_o = True
        elif board[3] == -1 and board[4] == -1 and board[5] == -1:
            lose_o = True
        elif board[6] == -1 and board[7] == -1 and board[8] == -1:
            lose_o = True
        #horz
        elif board[1] == -1 and board[4] == -1 and board[7] == -1:
            lose_o = True
        elif board[0] == -1 and board[3] == -1 and board[6] == -1:
            lose_o = True
        elif board[2] == -1 and board[5] == -1 and board[8] == -1:
            lose_o = True
        #dgonal
        elif board[0] == -1 and board[4] == -1 and board[8] == -1:
            lose_o = True
        elif board[2] == -1 and board[4] == -1 and board[6] == -1:
            lose_o = True
 
        b = 0
        for c in board:
            if c == 0:
                b += 1

        reward = 0
        if lose_o == True:
            reward = -LOSE_PENALTY
        elif won_o:
            reward = WON_REWARD
        elif b == 0:
        	reward = DRAW_REWARD
        ## NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        new_obs = tuple(board)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == WON_REWARD:
            new_q = WON_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        episode_reward += reward
        if reward == WON_REWARD or reward == -LOSE_PENALTY or reward == DRAW_REWARD:
            break

    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    print(episode,"/",HM_EPISODES,"done",end="\r")

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable/qtable-{int(time.time())}-ttc.pickle", "wb") as f:
    pickle.dump(q_table, f)
