import numpy as np
import matplotlib.pyplot as plt

# Global dictionaries for robot movement and sensing as in tester.py
dir_sensors = {'u': ['l', 'u', 'r'], 'r': ['u', 'r', 'd'],
               'd': ['r', 'd', 'l'], 'l': ['d', 'l', 'u'],
               'up': ['l', 'u', 'r'], 'right': ['u', 'r', 'd'],
               'down': ['r', 'd', 'l'], 'left': ['d', 'l', 'u']}
dir_move = {'u': [0, 1], 'r': [1, 0], 'd': [0, -1], 'l': [-1, 0],
            'up': [0, 1], 'right': [1, 0], 'down': [0, -1], 'left': [-1, 0]}
dir_reverse = {'u': 'd', 'r': 'l', 'd': 'u', 'l': 'r',
               'up': 'd', 'right': 'l', 'down': 'u', 'left': 'r'}

# I chose the coordinates as follow
delta = [[0, 1 ], # go up
[ -1, 0], # go left
[ 0, -1 ], # go down
[ 1, 0]] # go right
# Though the robot has three sensors facing three directions out of four at a time,
# all four directions will be sensed depending on robot headings
delta_degrees = {'up': 0,'left': -90,'right':90, 'down':180,'u': 0,'l': -90,'r':90, 'd':180}

delta_name = ['u','l','d','r']

path_symbol = {'up': 'v','left': '<','right':'>', 'down':'^','u': 'v','l': '<','r':'>', 'd':'^'}



class Robot(object):
    def __init__(self, maze_dim):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''

        self.location = [0, 0]
        self.heading = 'up'
        self.maze_dim = maze_dim
        
        # Number for wall description at each maze.
        self.maze_grid = np.zeros((maze_dim, maze_dim), dtype=np.int)
        # the number of the repeated path
        self.path_grid = np.zeros((maze_dim, maze_dim), dtype=np.int)
        self.exploration_path_symbol_grid = [[' ' for _ in range(maze_dim)] for _ in range(maze_dim)]   # optimal path direction with symbol
        
        self.exploration_path_grid = [[' ' for _ in range(maze_dim)] for _ in range(maze_dim)]      # optimal path direction with 'up','left', .... 
        
        self.exploration_path_value = [[' ' for _ in range(maze_dim)] for _ in range(maze_dim)]
        self.path_value = [[99 for _ in range(maze_dim)] for _ in range(maze_dim)] # value assigned from goal to start
        self.optimal_policy = [[' ' for _ in range(maze_dim)] for _ in range(maze_dim)]
        
    
        self.goal_bound = [maze_dim/2 - 1, maze_dim/2]  # destination area  - center of the maze

        self.heuristic = self.make_heuristic_grid(maze_dim) 

        self.back = 0  # Initially, the back of the robot is closed.
        self.count = 0
        self.run = 0
        
            
        
        
    def next_move(self, sensors):
        '''
        Use this function to determine the next move the robot should make,
        based on the input from the sensors after its previous move. Sensor
        inputs are a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.

        Outputs should be a tuple of two values. The first value indicates
        robot rotation (if any), as a number: 0 for no rotation, +90 for a
        90-degree rotation clockwise, and -90 for a 90-degree rotation
        counterclockwise. Other values will result in no rotation. The second
        value indicates robot movement, and the robot will attempt to move the
        number of indicated squares: a positive number indicates forwards
        movement, while a negative number indicates backwards movement. The
        robot may move a maximum of three units per turn. Any excess movement
        is ignored.

        If the robot wants to end a run (e.g. during the first training run in
        the maze) then returing the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        '''
        
        if self.run == 0 :   # Define robot's movement during the training run
             rotation, movement = self.exploration_phase(sensors)
             
        elif self.run == 1:  # Define robot's movement during the test run
            rotation, movement = self.exploitation_phase(sensors)
            
            
        
        return rotation, movement
        
    def maze_number(self, sensors, back):
        # Compute the wall number that describes which edges are closed or open at each position from the robot's sensors
        
        # Scale sensor reading to open/close (1 for open or 0 for closed)
        for i in range(len(sensors)):
            if sensors[i] > 0 :
                sensors[i] = 1
                
        # The wall number at each position upon sensor headings        
        if self.heading == 'u' or self.heading == 'up':
            number = sensors[1] + 2*sensors[2] + 4*back + 8*sensors[0]
        elif self.heading =='d' or self.heading == 'down':
            number = back + 2*sensors[0] + 4*sensors[1] + 8*sensors[2]
        elif self.heading == 'l' or self.heading == 'left':
            number = sensors[2] + 2*back + 4*sensors[0] + 8*sensors[1]
        else:
            number = sensors[0] + 2*sensors[1] + 4*sensors[2] + 8*back
        
        return number

    def update_robot_heading_location(self, rotation, movement):
        # Update robot heading and location
    
        # Convert from robot turn angles, [-90, 0, 90]  to [0, 1, 2]
        rotation_index = rotation/90 + 1
        
        # Compute new heading from the robot heading rotation
        self.heading  = dir_sensors[self.heading][rotation_index]
        
        # Update location with regard to heading direction
        self.location[0] += dir_move[self.heading][0]*movement
        self.location[1] += dir_move[self.heading][1]*movement

    def make_heuristic_grid(self,maze_dim):
        # define heuristic function for any maze layout. i.e. it is only based on  the distance goal. 
        # Assign 0 at the center and incrementally higher number as span out away from the center.
        

        # Intialized with zeros
        heuristic = np.zeros((maze_dim,maze_dim), dtype=np.int)
        
        # Let's suppose a vertical axis crossed any given maze vertically in half (thus maze_dim/2) and,
        # similarly, a horizontal axis crossed the same maze horizontally dividing the maze into four quadrants
        # The center maze being 2 by 2, one by one center square (the goal with number 0) will be for each of the four quadrants 
        
        # consider top-right quadrant section
        # counting 0 at upper-right center square of the intersection of the axis 1, 2, 3 and so on for the following squares on same row and
        # column out wards 
        # top rght quadran
        for i in range(maze_dim/2):
            for j in range(maze_dim/2,maze_dim):
                heuristic[i,j] = j -maze_dim/2 + abs(i+1 -maze_dim/2)
        
        # similarly, assign top-left quadrant section
        for i in range(maze_dim/2,maze_dim):
            for j in range(maze_dim/2,maze_dim ):
                heuristic[i,j] = i+j - maze_dim
        
        
        # consider bottom-left quadrant section 
        for i in range(maze_dim/2):
            for j in range(maze_dim/2):
                heuristic[i,j] = abs(i+1 -maze_dim/2) + abs(j+1 -maze_dim/2)
        
        # assign numbers to bottom-right quadrant section         
        for i in range(maze_dim/2, maze_dim):
            for j in range(maze_dim/2):
                heuristic[i,j] = i -maze_dim/2 +abs(j+1 - maze_dim/2)
                
        
       
                
        return heuristic 
        
    
    def plot_robot_path(self,x1,y1,x2,y2,number,i):
        # Plot each step path and display the assigned step number.
    
        plt.figure(i)    
        plt.plot([x1,x2],[y1,y2],'b')
        plt.plot([x1],[y1],marker = '$'+str(number)+'$', markersize =20)
        plt.axis([ -1, self.maze_dim, -1, self.maze_dim])
        plt.grid(True)
        

    def compute_next_step(self,x1,y1,sensors):
        # Compute next step for the robot
        
        robot_turn_angl = [-90, 0, 90]
        open = []
               
        if sensors == [0,0,0]:
            # meaning there is a wall at each of the three directions, therefore, the robot shouldn't move to hit the wall but just
            # turn 90 degrees if the robot find itself in a dead end maze cell
            # It could also turn -90 instead 
            movement = 0
            rotation = 90
            print "\n### Robot entered a maze cell with no other way out but to turn back out through the open entrance ###  \n"
            
        else:  #  Move to open position   
            for i in range(3): # Check all the open sensor directions
                if sensors[i] > 0 : 
                    sensors[i] = 1  # Scale sensor reading to open/close (1 or 0) 
                    
                    # Select next position in the direction of open wall
                    x2 = x1 + dir_move[dir_sensors[self.heading][i]][0]   
                    y2 = y1 + dir_move[dir_sensors[self.heading][i]][1] 
                    
                    if x2>=0 and x2<self.maze_dim and y2>=0 and y2<self.maze_dim:    # Check if x2, y2 are inside the maze
                        t2 = self.path_grid[x2][y2]  # the number  times travelled on the path  - 0 means "never been travelled on the path"
                        h2 = self.heuristic[x2][y2]  # distance to the center - smaller number means "closer to the center"
                        open.append([t2,h2,x2,y2,i]) # make a list of possible paths
                        
            next = min(open)  # Pick the path that is less travelled and closer to center (the goal)  
            t,h,x,y,i = next  # Unpack the picked path
            rotation,movement = robot_turn_angl[i], 1  # Assign rotation direction and movement = 1 for exploration phase

        return rotation, movement                    
         

    def compute_value_function(self, goal):
        # Compute value function based on the obtained map.
        # This is a dynamic programming method used to find the optimal path in Artificial Intelligence For Robotics course.
    
        change = True   # Intialized a boolean variable to start a while-loop   
        
        while change:
            change = False  # Stop while-loop if no change happens in the for-loop below
            
            # Assign 0 to the goal position and span out incrementally wherever wall is open.
            for x in range(self.maze_dim):
                for y in range(self.maze_dim):
                    # assign zero to the goal position and continue                                         
                    if goal[0] == x and goal[1] == y:                        
                        if self.path_value[x][y] > 0:  # Just to be explicit
                            self.path_value[x][y] = 0
                            self.optimal_policy[x][y] = '*'
                            print "Found the Goal!!!! for the first run"
                            
                            print "The second run result is coming up in a minute."  
                            change = True
                            
                            # Plot '*' on the goal position
                            plt.figure(9)    
                            plt.plot([x],[y], marker = '$'+ str(0) + '$', markersize = 20)
                            plt.axis([ -1, self.maze_dim, -1, self.maze_dim])
                            plt.grid(True)
                            # Plot '*' on the goal position
                            plt.figure(8)    
                            plt.plot([x],[y], marker = '*', markersize = 20, color = 'b')
                            plt.axis([ -1, self.maze_dim, -1, self.maze_dim])
                            plt.grid(True)
                            plt.title("The goal position")
                            plt.savefig("my goal position.png", format="png")
                            
                            

                    
                    # Increase the value function by 1 adjacent to  the position that changed in the loop
                    else:
                        
                        # Read the maze map; decode the wall number into 4 digits binary; and check which direction is open;  
                        wall_number = self.maze_grid[x][y]
                        wall_binary = bin(wall_number)[2:].zfill(4)
        
                        left = wall_binary[0]
                        down = wall_binary[1]
                        right = wall_binary[2]
                        up = wall_binary[3]

                        # Expand the value function to open wall direction
                        for a in range(len(delta)): 
                            if (up == '1' and a == 0) or (left == '1' and a ==1) or (down == '1' and a==2) or (right == '1' and a == 3):

                                x2 = x + delta[a][0]
                                y2 = y + delta[a][1]

                                if x2 >= 0 and x2 < self.maze_dim and y2 >= 0 and y2 < self.maze_dim:    # Check if it is inside of maze
                                    v2 = self.path_value[x2][y2] + 1   # Increase the value function by 1
                                    
                                    if v2 < self.path_value[x][y]: # Update value function if it was changed previously
                                        change = True
                                        self.path_value[x][y] = v2
                                        self.optimal_policy[x][y] = delta_name[a]  # Update the optimal_policy with the direction of movement
                                        
                                        # Plot value function
                                        plt.figure(9)    
                                        plt.plot([x],[y], marker = '$'+ str(v2) + '$', markersize = 20)
                                        plt.title("My value function")
                                        plt.annotate(xy=[0, 0], s='initial')
                                        plt.annotate(xy=self.goal_bound, s='goal')
                                        plt.savefig("my_value_function.png", format="png")
                                        

                                        # Plot optimal policy
                                        plt.figure(8)    
                                        plt.plot([x],[y], marker = '$' + delta_name[a] + '$', markersize = 20, color = 'b')
                                        plt.title("My optimal policy")
                                        plt.annotate(xy=[0, 0], s='initial')
                                        plt.annotate(xy=self.goal_bound, s='goal')
                                        plt.xlabel('r = right, l = left')
                                        plt.ylabel('d = down, u = up')
                                        plt.savefig("my_optimal_policy.png", format='png')


    def plot_value(x,y,number,i):
        plt.figure(i)    
        plt.plot([x+0.5],[y+0.5], marker = '$' + str(number) + '$', markersize = 20)
        plt.axis([ -1, 14, -1, 14])
        plt.grid(True)
  
              
    def exploration_phase(self,sensors): 
        # Run training session
        #print "heuristic"
        #print self.make_heuristic_grid(self.maze_dim)

          # print out counts: used for debugging
        print "exploration step Count: ", self.count,  sensors
        self.count +=1

        # Get robot's current position          
        x1 = self.location[0]
        y1 = self.location[1]

        # Add 1 as the robot passes position [x1,y1]
        self.path_grid[x1][y1] +=1  
        
        # Construct maze map with the data obtained from robot's sensors      
        number = self.maze_number(sensors, self.back) 
        self.maze_grid[x1][y1] = number
        
 
        # Select robot's next step
        rotation, movement = self.compute_next_step(x1, y1, sensors)
 
        # Update the robot's back wall condition (open:'1', closed:'0')
        self.back = 0 if movement == 0 else 1
        
        # Update robot heading and location after the next step
        self.update_robot_heading_location(rotation, movement)
        
        # Get robot's position after movement
        x2 = self.location[0]
        y2 = self.location[1]

        # Update the training path
        self.exploration_path_symbol_grid[x1][y1] = path_symbol[self.heading] 
        self.exploration_path_grid[x1][y1] = self.heading
       
        
        # Plot the maze map (wall number) along the path
        self.plot_robot_path(x1, y1, x2, y2, number, 1)
        plt.figure(13)
        plt.plot([x1,x2],[y1,y2],'b')
        plt.plot([x1],[y1],marker = '$'+str(number)+'$', markersize =20)
        plt.axis([ -1, self.maze_dim, -1, self.maze_dim])
        plt.grid(True)
        plt.savefig("My_maze_map.png", format="png")
        
        
        # Plot the number of steps taken along the path
        self.plot_robot_path(x1, y1, x2, y2, self.count, 2)
        plt.figure(14)
        plt.plot([x1,x2],[y1,y2],'b')
        plt.plot([x1],[y1],marker = '$'+str(self.count)+'$', markersize =20)
        plt.axis([ -1, self.maze_dim, -1, self.maze_dim])
        plt.grid(True)
        plt.savefig("robot_step_number.png", format="png")
        
        # Chech if it reached the goal
        if x1 in self.goal_bound and y1 in self.goal_bound:
            print "\n*** Robot has successfully reached the goal position after {} steps!!!!  *** \n".format(self.count)
            #print "exploration phase with symbols indicating the robot's move directions while exploring the cells of the maze"
            #print self.exploration_path_symbol_grid
            
            #print "The path taken by the robot before reaching the goal"
            #print self.exploration_path_grid
            
            #print "the maze grid, 0 indicates an unexplored maze cell, 1 indicates one open and three walls and so on"
            #print self.maze_grid
            
            # Compute value function and optimal policy for the maze 
            goal = [x1,y1]
            self.compute_value_function(goal)
            #print "optimal policy"
            #print self.optimal_policy
            
            # Reset for test run
            rotation = 'Reset'
            movement = 'Reset'
            self.run = 1
            self.location = [0,0]
            self.heading = 'up'
            self.count = 0
            
        return rotation, movement

    def exploitation_phase(self,sensors):
        # Run test session
    
        # print out counts: used for debugging
        print "exploitation step Count: ", self.count, sensors
        self.count +=1

           
        movement = 1 # Set default movement = 1
        
        # Get robot's current position 
        x1 = self.location[0]
        y1 = self.location[1]
            
        
        # Determine angle rotation comparing the current heading and  optimal path direction
        robot_heading_degrees = delta_degrees[self.heading]
        robot_optimal_heading_degrees = delta_degrees[self.optimal_policy[x1][y1]]
        rotation = robot_optimal_heading_degrees - robot_heading_degrees 
        
        # Make sure rotation to be -90, 0, 90
        if rotation == -270:
            rotation = 90
        elif rotation == 270:
            rotation = -90
            
        # Convert from rotatio angles, [-90, 0, 90]  to [0, 1, 2]    
        rotation_index = rotation/90 + 1  
        direction  = dir_sensors[self.heading][rotation_index]  # Compute new headig from the rotation
        
        #  Take  upto 3 steps at once if heading is the same
        x,y = x1,y1
        while(movement<3):
            current = self.optimal_policy[x][y]
            x += dir_move[direction][0]
            y += dir_move[direction][1] 

            if self.optimal_policy[x][y] == current:
                movement += 1           

            else: 
                break
        # Update robot's position
        self.update_robot_heading_location(rotation, movement)
        
        # Get robot's position after movement
        x2 = self.location[0]
        y2 = self.location[1]
    
        # Plot optimal path
        plt.figure(4) 
        plt.plot([x1,x2],[y1,y2],'b')
        plt.axis([ -1, self.maze_dim, -1,self.maze_dim])
        plt.grid(True)
        plt.title("The optimal path")
        plt.annotate(xy=[0, 0], s='initial')
        plt.annotate(xy=self.goal_bound, s='goal')
        plt.savefig("my_Optimal_path.png", format='png')
       
        return rotation, movement
        
        