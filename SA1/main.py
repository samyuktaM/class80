import pygame,math
import neat

import pickle

from helper import getSensorX, getSensorY

pygame.init()
screen = pygame.display.set_mode((1067,600))

pygame.display.set_caption("Car racing")
background_image = pygame.image.load("track.png").convert()
player_image = pygame.image.load("car.png").convert_alpha()

player=pygame.Rect(400,300,20,20)

WHITE=(255,255,255)
xvel=2
yvel=3
angle=0
change=0

distance=2
forward=False

font = pygame.font.Font('freesansbold.ttf', 12)

# Define save(winner) function
def save(winner):
    # Create file_name variable and save 'std1.pkl' name in it
    file_name = "std1.pkl"
    # Open the file in file_name in "wb" mode
    with open(file_name, "wb") as file:
        # Use pickel.dump to save the winner in the file
        pickle.dump(winner, file)
        # Print object successfully saved to file_name
        print(f'object successfully saved to "{file_name}"')

def newxy(x,y,distance,angle):
  angle=math.radians(angle+90)

  xnew=x+(distance*math.cos(angle))
  ynew=y-(distance*math.sin(angle))

  return xnew,ynew

def checkOutOfBounds(car):
  x = car.x
  y = car.y
  width = car.width
  height = car.height

  if(checkPixel(x,y) or checkPixel(x+width, y) or checkPixel(x, y+height) or checkPixel(x+width, y+height)):
      return True
  
def checkPixel(x, y):
    global screen
    try:
        color = screen.get_at((x, y))
    except:
        return 1
    if(color == (137,137,137,255)):
        return 0
    return 1

def getSensorsData(car, angle):
    global screen
    margin = 55
    
    x = car.x + car.w/2
    y = car.y 
    
    sensorAngles = [-10,-30,-50,-70,-90,-110,-130,-150,-170]
    sensorData= []

    for sensorAngle in sensorAngles:
        newX = getSensorX(x, angle, sensorAngle, margin)
        newY = getSensorY(y, angle, sensorAngle, margin)      
        sensorData.append(checkPixel(newX, newY))
        pygame.draw.rect(screen,(0, 255,0), [newX, newY, 5, 5])
        
    return sensorData
    
gen=0
angle =0

def eval_fitness(generation, config):
    global angle, gen, forward, change
    gen = gen+1
    genomeCount = 1

    print("Generation:", gen, "Total", len(generation) )
    
    for gid, genome in generation:
        
        genome.fitness = 0 
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        infoText = font.render('Generation'+ str(gen)+  ' genomecount: (' +str(len(generation))+") :" +str(genomeCount), True, (255,255,0))

        while True:
          screen.blit(background_image,[0,0])        
          pygame.draw.rect(screen,(0,0,0), [400, 0, 210, 78])
          screen.blit(infoText, (402, 5))
          
          fitnessText = font.render('fitness Score:'+ str(genome.fitness) , True, (255,255,0))
          screen.blit(fitnessText, (402, 65))
          
          for event in pygame.event.get():
            if event.type == pygame.QUIT:
              pygame.quit()
              
            if event.type == pygame.KEYDOWN:
               if event.key == pygame.K_LEFT:
                  change = 5
               if event.key ==pygame.K_RIGHT:
                change = -5 

               if event.key == pygame.K_UP:
                forward = True
                
            if event.type == pygame.KEYUP:
              if event.key ==pygame.K_LEFT or event.key == pygame.K_RIGHT:
                  change = 0
              if event.key == pygame.K_UP:
                forward = False 
            
         
          if forward:
              player.x,player.y=newxy(player.x, player.y, 3, angle)  
                          
          if(checkOutOfBounds(player)):
              player.x = 170
              player.y = 300
              angle =0
              genomeCount = genomeCount +1
              break
          
          angle = angle + change
          
          newimage=pygame.transform.rotate(player_image,angle) 
          pygame.draw.rect(screen,(0, 255, 0), player)
          screen.blit(newimage ,player)
            
          forward = True
          change = 0
          
          sensorData= getSensorsData(player, angle)     
          print(sensorData)

          output = net.activate((sensorData[0], sensorData[1], sensorData[2], sensorData[3], sensorData[4], sensorData[5], sensorData[6], sensorData[7], sensorData[8]))
          
          inputText = font.render('All Sensors:'+ str(sensorData) , True, (255,255,0))
          screen.blit(inputText, (402, 20))
          output1Text = font.render('Output1:'+ str(output[0]), True, (255,255,0))
          screen.blit(output1Text, (402, 35))
          output2Text = font.render('Output2:'+ str(output[1]), True, (255,255,0))
          screen.blit(output2Text, (402, 50))

          if output[0] > 0.65:
             change = 3
          if output[1] > 0.65:
             change = -3
                    
          genome.fitness += 0.2

          # If fitness of genome becomes > 1000 then save the genome and break the loop 
          if genome.fitness > 1000:
             save(genome)
          
          
          pygame.display.update()
          pygame.time.Clock().tick(30)
  
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,'config-feedforward.txt')  
p = neat.Population(config)
winner = p.run(eval_fitness,10) 

# Call save(winner) to save the best genome
save(winner)