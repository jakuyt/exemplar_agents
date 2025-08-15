import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ExemplarTheoryPy as et
from scipy.stats import multivariate_normal

# Initialize parameters
n_time_steps = 1000000
environmental_bias = [0, 0]
noise_std = 40   # in Hz
decay_rate = 0.01
weight_threshold = 0.5
cat_size = 50
frame_interval = 10

# Initialize agents
agents = []
agent = et.Agent('Alice')
for vowel in ['i', 'e', 'a', 'o', 'u']:
  cat = et.Category(vowel, mean=et.vowel_data[vowel], num_exemplars=cat_size)
  agent.add_category(cat)
agents.append(agent)
agent = et.Agent('Bob')
for vowel in ['i', 'e', 'a', 'o', 'u']:
  cat = et.Category(vowel, mean=et.vowel_data[vowel], num_exemplars=cat_size)
  agent.add_category(cat)
agents.append(agent)

# Create a figure and axis
fig, axs = plt.subplots(1,len(agents))

model = et.Environment(agents, axs)

# Create the animation
ani = animation.FuncAnimation(
    fig,
    init_func=model.initialize_fig,
    func=model.update_exemplars,
    frames=n_time_steps,
    blit=False,
    interval=1,
    cache_frame_data=False,  # Disable frame caching
    repeat=False,
    fargs=(decay_rate, weight_threshold, environmental_bias, noise_std)
)

# Display the animation
plt.rcParams['animation.html'] = 'html5'
fig.set_dpi(80)  # Lower DPI for faster rendering
plt.gcf().set_size_inches(12, 6) 
plt.show()

#Diary #4
#Worked on the classes and functions. Put them in a separate file. Also kinda figured out that my code shouldn't have worked? But it did? And now I'm trying to fix it and it doesn't work.
#This animation functions is kinda hard to understand. You want to keep updating the agents but also the update function shouldn't take any variables. So it's so tough to get those agents in.

#Diary #5
#The animation now works. But the gapping between the categories does not yet emerge, I think I have to change the way the tokens are categories or discarded. Played around with a variant where
#each new token is added with a probability p equal to the confidence score of the category with the highest condifence score. That means that if two categories have equal confidence score p=0.5
#the token is added to a category with p=1-0.5=50% likelihood. Otherwise, it is discarded. If there is a clear winner, lets say confidence = 0.9, then the token is added to that category with p=0.9 and
#discarded with p=1-0.9=10%.

#I tried this. But still the desired behaviour does not emerge. Truly curious. I will have to think about this some more. Perhaps the code does not behave the way I intended. Or perhaps what I misunderstand
#why the current behaviour persists.

#Diary #6
#The fix was to normalize the softmin function I used to decide which category the token should be added to. The only problem now is technical. Since I need to only consider the best two candidates and not the rest.
#Since the distance to the other categories might warp the normalization. Resulting in weird behaviour.
#I'm trying something using the collections package. I think I'll leave this for another day.

#Diary #7
#I fixed the categorization algorithm by keeping track of the indices. I did not need a paired list. I also fixed something I didn't know was wrong lol. Apparently when creating a new exemplar I was also updating the
#exemplar it was modeled after.

#Diary #8
#Today I want to add the colourmap and a trail of the mean of every category.

#Diary #9
#Added the colormap, a trail for the mean and some text displaying the number of exemplars in the category and the weight. I experimented with the decay rate and the weight threshold.
#I also experimented with a chain shift. Which does seem to emerge right now. I still want to tweak how new exemplars are categorized. Right now it seems a bit convoluted.
#It seems that a higher decay rate increases the speed at which the shift happens, which is as expected.
#I also want to add a way to account for the range of human hearing/production.