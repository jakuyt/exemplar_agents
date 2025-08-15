import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os

range_F1 = [150, 950]
range_F2 = [400, 2600]
diff_F1 = range_F1[1] - range_F1[0]
diff_F2 = range_F2[1] - range_F2[0]
rate_F2_F1 = diff_F2 / diff_F1

vowel_data = {
    'i': [240, 2400],
    'y': [235, 2100],
    'e': [390, 2300],
    'ø': [370, 1900],
    'ɛ': [610, 1900],
    'œ': [590, 1700],
    'æ': [585, 1710],
    'a': [850, 1610],
    'ɶ': [820, 1530],
    'ɑ': [750, 940],
    'ɒ': [700, 760],
    'ʌ': [600, 1170],
    'ɔ': [500, 700],
    'ɤ': [460, 1310],
    'o': [360, 640],
    'ɯ': [300, 1390],
    'u': [250, 595]
}

def softmax(x, temp = 1):
    # Subtract the max value for numerical stability
    e = np.exp(temp * (x - np.max(x)))
    return e / e.sum()

def softmin(x, temp = 1):
    #Subtract the min value for numerical stability
    e = np.exp(-(x - np.min(x))/temp**2)
    return e / e.sum()

def normalized_softmin(x, temp = 1):
    #normalize the vector
    x = x / np.linalg.norm(x)
    #Subtract the min value for numerical stability
    e = np.exp(-(x - np.min(x))/temp**2)
    return e / e.sum()

class Exemplar:
    def __init__(self, pos, weight):
        if len(pos) != 2:
            raise ValueError("Uhmm actually, the position variable is 2d. It is defined as (f1, f2).")
        self.pos = np.array(pos)
        self.weight = weight

    def is_within_range(exemplar):
        """Checks if an exemplar's position is within the specified F1 and F2 ranges.

        Args:
          exemplar: An Exemplar object.
          range_F1: A tuple representing the (min, max) values for F1.
          range_F2: A tuple representing the (min, max) values for F2.

        Returns:
          True if the exemplar's position is within both ranges, False otherwise.
        """
        F1, F2 = exemplar.pos  # Get F1 and F2 values from the exemplar
        return range_F1[0] <= F1 <= range_F1[1] and range_F2[0] <= F2 <= range_F2[1]

    def __repr__(self):
        """
        Provide a string representation of the Exemplar instance.
        """
        return f"Exemplar(position={self.pos}, weight={self.weight})"

class Category:
    def __init__(self, name, mean=[600, 1500], covariance=[[1000, 0], [0, 4000]], num_exemplars=0, weight = 1.0):
        """
        Initialize a Category instance.

        :param name: The name of the category.
        :param mean: Mean of the Gaussian distribution for generating exemplars (optional).
        :param covariance: Covariance matrix for the Gaussian distribution (optional).
        :param num_exemplars: Number of exemplars to generate in the standard cloud (optional).
        """
        self.name = name
        self.exemplars = []

        # Generate a cloud of exemplars if parameters are provided
        if mean is not None and covariance is not None and num_exemplars > 0:
            self._generate_exemplars(mean, covariance, num_exemplars, weight)

    def _generate_exemplars(self, mean, covariance, num_exemplars, weight):
        """
        Generate a cloud of exemplars based on the given mean and covariance.

        :param mean: Mean of the Gaussian distribution (list or array of length 2).
        :param covariance: Covariance matrix for the Gaussian distribution (2x2 matrix).
        :param num_exemplars: Number of exemplars to generate.
        """
        # Generate random samples from a multivariate normal distribution
        positions = np.random.multivariate_normal(mean, covariance, num_exemplars)

        # Create exemplars and add them to the category
        for position in positions:
            self.add_exemplar(Exemplar(position, weight=np.random.uniform(0, 1)))  # Default weight is 1.0

    def add_exemplar(self, exemplar):
        """
        Add an Exemplar to the category.

        :param exemplar: An instance of the Exemplar class.
        """
        if not isinstance(exemplar, Exemplar):
            raise TypeError("Only objects of type 'Exemplar' can be added.")
        if Exemplar.is_within_range(exemplar):
            self.exemplars.append(exemplar)
        else:
            exemplar.pos[0] = np.clip(exemplar.pos[0], range_F1[0], range_F1[1])
            exemplar.pos[1] = np.clip(exemplar.pos[1], range_F2[0], range_F2[1])
            self.exemplars.append(exemplar)

    def remove_exemplar(self, exemplar):
        """
        Remove an Exemplar from the category.

        :param exemplar: The Exemplar instance to remove.
        """
        if exemplar in self.exemplars:
            self.exemplars.remove(exemplar)
        else:
            print(f"Exemplar not found in the category. Target category: {self}. Target exemplar: {exemplar}")

    def get_positions(self):
        """
        Get the positions of all exemplars in the category.

        :return: A list of numpy arrays representing the positions of exemplars.
        """
        positions = [exemplar.pos for exemplar in self.exemplars]
        if not positions:
            print("Category is empty. Cannot get positions.")
            return None  # Return None if there are no exemplars

        return positions

    def get_weights(self):
        """
        Get the weights of all exemplars in the category.

        :return: A list of floats representing the weights of exemplars.
        """
        weights = [exemplar.weight for exemplar in self.exemplars]
        if not weights:
            print("Category is empty. Cannot get weights.")
            return None  # Return None if there are no exemplars

        return weights

    def get_weighted_mean(self):
        """
        Calculate the weighted mean of the exemplars in the category.

        :return: A numpy array representing the weighted mean.
        """

        if not self.exemplars:
            print("Category is empty. Cannot calculate weighted mean.")
            return None  # Return None if there are no exemplars

        weighted_sum = np.zeros(2)  # Initialize a 2D vector for the sum
        total_weight = 0

        for exemplar in self.exemplars:
            weighted_sum += exemplar.pos * exemplar.weight  # Weighted sum of positions
            total_weight += exemplar.weight  # Total weight

        return weighted_sum / total_weight  # Weighted average

    def get_weighted_var(self):
        """
        Calculate the weighted variance of the exemplars in the category.

        :return: A numpy array representing the weighted variance.
        """
        if not self.exemplars:
            print("Category is empty. Cannot calculate weighted variance.")
            return None  # Return None if there are no exemplars

        pos = [exm.pos for exm in self.exemplars]
        weights = [exm.weight for exm in self.exemplars]

        # Calculate weighted covariance
        weighted_var = np.cov(pos, rowvar=False, aweights=weights)

        return weighted_var

    def get_weighted_random_exemplar(self):
        """
        Get a random exemplar from the category.

        :return: A random exemplar from the category.
        """

        if not self.exemplars:
            print("Category is empty. Cannot get a random exemplar.")
            return None  # Return None if there are no exemplars

        exemplar_index = np.random.choice(
          a = len(self.exemplars),
          p = softmax(self.get_weights())
        )
        random_exemplar = self.exemplars[exemplar_index]
        return random_exemplar

    def __repr__(self):
        """
        Provide a string representation of the Category instance.
        """
        return f"Category(name={self.name}, exemplars={len(self.exemplars)})"

class Agent:
    def __init__(self, name):
        """
        Initialize an Agent instance.

        :param name: The name of the agent.
        """
        self.name = name
        self.categories = []  # List to store Category objects

    def add_category(self, category):
        """
        Add a Category to the agent.

        :param category: An instance of the Category class.
        """
        if not isinstance(category, Category):
            raise TypeError("Only objects of type 'Category' can be added.")
        self.categories.append(category)

        return self

    def remove_category(self, category):
        """
        Remove a Category from the agent.

        :param category: The Category instance to remove.
        """
        if category in self.categories:
            self.categories.remove(category)
        else:
            raise ValueError("Category not found in the agent.")

        return self

    def display_categories(self):
        """
        Display the categories associated with the agent.
        """
        global range_F1, range_F2

        # Set up the plot
        fig, ax = plt.subplots()
        ax.set_ylim(*range_F1)  # Adjust x-limit as needed
        ax.set_xlim(*range_F2)  # Adjust y-limit as needed
        ax.set_ylabel("F1")
        ax.set_xlabel("F2")

        # Move the F1 axis to the top and F2 to the right
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()

        # Invert the axes
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_title("Linguistic Exemplar Dynamics (2D)")

        for cat in self.categories:
          # Generate scatterplot
          # First we prep the exemplar list
          y = [exm.pos[0] for exm in cat.exemplars]
          x = [exm.pos[1] for exm in cat.exemplars]

          # Then we plot
          ax.scatter(x, y, s=1, label = cat.name)
          ax.legend()

    def __repr__(self):
        """
        Provide a string representation of the Agent instance.
        """
        return (
            f"Agent(name={self.name}, categories={[cat.name for cat in self.categories]})"
        )
    
class Environment:
    def __init__(self, agents, axs):
        self.agents = agents
        self.axs = axs
        self.scatterplots = []
        self.mean_history = {agent.name: {cat.name: {'x': [], 'y': []} 
                           for cat in agent.categories} 
                           for agent in agents}
        self.mean_lines = []
        self.text_annotations = []

    def update_exemplars(self, frame, decay_rate, weight_threshold, environmental_bias, noise_std):
        
        """
        Update exemplars for each agent in the environment.

        This function simulates the interaction between agents by updating the weights
        of exemplars, adding noise and bias, and transferring exemplars between agents.
        It then updates the plot to reflect the changes in exemplar positions.

        Args:
            frame: The current frame of the animation (unused).
            decay_rate: The rate at which exemplar weights decay.
            weight_threshold: The threshold below which exemplars are removed.
            environmental_bias: A bias vector added to exemplar positions.
            noise_std: The standard deviation of the Gaussian noise added to exemplar positions.

        Returns:
            The updated axes objects for the animation.
        """

        # Choose a random agent
        speaking_agent = np.random.choice(self.agents)
        listening_agent = np.random.choice(self.agents)

        # Generate new token by choosing random exemplar and give it w = 1.0
        random_cat = np.random.choice(speaking_agent.categories)
        model_exm = random_cat.get_weighted_random_exemplar()
        new_token = Exemplar(model_exm.pos, 1)
        # Model a sound change where y is fronted. This is done by adding 100 Hz to the F2 value of u vowels. 
        if random_cat.name == 'y':
            p = min(1, 0.90)
            new_token.pos = p*new_token.pos + (1-p)*np.array([*vowel_data['i']])

        # Add gaussian noise and environmental bias to new exemplar
        # Noise is between 1 and 100 kHz
        gauss_noise = np.random.multivariate_normal(mean=[0,0] , cov=noise_std*np.array([[1,0],[0,diff_F2/diff_F1]]))
        new_token.pos += gauss_noise + environmental_bias

        # Get means of listening agent categories to test new exemplars against
        dist_to_cats = []
        for i, cat in enumerate(listening_agent.categories, start=0):
            cat_mean = cat.get_weighted_mean()
            exp_dist = float(np.exp(-np.linalg.norm(cat_mean - new_token.pos)/125))
            dist_to_cats.append(exp_dist)

        # Get the two closest categories
        # Ensure there are at least two distances to partition
        if len(dist_to_cats) > 1:
            indices = np.argpartition(dist_to_cats, -2)[:2]
        else:
            indices = [0]
        best_cands = []
        best_cands_dist = []
        for i in indices:
            best_cands.append(listening_agent.categories[i])
            best_cands_dist.append(dist_to_cats[i])

        # Choose the closest category (if any) as target category and add exemplar
        cand_confidence_scores = normalized_softmin(best_cands_dist, temp=0.5)

        # Add exemplar to target category if confidence is high enough
        x = np.random.rand()
        target_cat = best_cands[np.argmax(cand_confidence_scores)]

        if x < np.max(cand_confidence_scores):
            target_cat.add_exemplar(new_token)

            # Decay exemplar weights of target category
            for exemplar in target_cat.exemplars:
                exemplar.weight *= min(1,1-decay_rate)
                if exemplar.weight < weight_threshold:
                    target_cat.remove_exemplar(exemplar)

        #If confidence is not high enough, do not add exemplar

        # Update scatterplots
        for i, agent in enumerate(self.agents, start=0):

            for j, cat in enumerate(agent.categories, start=0):
                # Generate scatterplot
                # First we prep the exemplar list
                l = len(agent.categories)

                y = [exm.pos[0] for exm in cat.exemplars]
                x = [exm.pos[1] for exm in cat.exemplars]

                # Normalize weights for alpha values
                weights = [exm.weight for exm in cat.exemplars]
                norm_weights = np.array(weights) / max(weights)

                # Create scatterplot and add to list
                self.scatterplots[i*l+j].set_offsets(np.c_[x, y])
                self.scatterplots[i*l+j].set_alpha(norm_weights)

        # After updating scatter plots, update mean traces
        for i, agent in enumerate(self.agents):
            for j, cat in enumerate(agent.categories):
                # Calculate current mean
                if len(cat.exemplars) > 0:
                    mean_pos = cat.get_weighted_mean()
                    
                    # Store in history
                    self.mean_history[agent.name][cat.name]['y'].append(mean_pos[0])
                    self.mean_history[agent.name][cat.name]['x'].append(mean_pos[1])
                    
                    # Update line
                    line_idx = i * len(agent.categories) + j
                    self.mean_lines[line_idx].set_data(
                        self.mean_history[agent.name][cat.name]['x'],
                        self.mean_history[agent.name][cat.name]['y']
                    )

        # Update text annotations with current statistics
        for i, agent in enumerate(self.agents):
            for j, cat in enumerate(agent.categories):
                if len(cat.exemplars) > 0:
                    mean_pos = cat.get_weighted_mean()
                    total_weight = np.sum([ex.weight for ex in cat.exemplars])
                    text = f'{cat.name}\nN={len(cat.exemplars)}\nW={total_weight:.1f}'
                    
                    # Update annotation position and text
                    ann_idx = i * len(agent.categories) + j
                    self.text_annotations[ann_idx].set_position((mean_pos[1], mean_pos[0]))
                    self.text_annotations[ann_idx].set_text(text)

        #os.system('cls' if os.name == 'nt' else 'clear') # Clear the terminal
        frame_data = {
            'frame': frame,
            'speaking_agent': speaking_agent.name,
            'listening_agent': listening_agent.name,
            'new_token_pos': new_token.pos,
            'best_cands': [cat.name for cat in best_cands],
            'distances': best_cands_dist,
            'conf_scores': cand_confidence_scores,
            'target_cat': target_cat.name,
            'number of exemplars': len(target_cat.exemplars),
            'total weight': np.sum(target_cat.get_weights())
        }
        #print(frame_data)
    
        # Return the updated artists to FuncAnimation
        return self.scatterplots + self.mean_lines + self.text_annotations

    def initialize_fig(self):

        """
        Initialize the figure and axes.

        This function sets up the plot with the correct limits, labels, titles, and
        inverts the axes. It also generates scatterplots of the exemplars in each
        category and adds legends.

        Returns:
            The updated axes objects.
        """
        
        for ax in self.axs:
            ax.set_ylim(range_F1[0], range_F1[1])  # Adjust x-limit as needed
            ax.set_xlim(range_F2[0], range_F2[1])  # Adjust y-limit as needed
            ax.set_ylabel("F1")
            ax.set_xlabel("F2")

            # Move the F1 axis to the top and F2 to the right
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()

            # Invert the axes
            ax.invert_yaxis()
            ax.invert_xaxis()
            ax.set_title("Linguistic Exemplar Dynamics (2D)")

        # Define a color map or list of colors
        colors = list(mcolors.TABLEAU_COLORS.values())

        for i, agent in enumerate(self.agents, start=0):
            # Grab the right axis
            ax = self.axs[i]

            for j, cat in enumerate(agent.categories):
                # Generate scatterplot
                # First we prep the exemplar list
                y = [exm.pos[0] for exm in cat.exemplars]
                x = [exm.pos[1] for exm in cat.exemplars]
                weights = [exm.weight for exm in cat.exemplars]  # Assuming each exemplar has a weight attribute

                # Normalize weights for alpha values
                norm_weights = np.array(weights) / max(weights)

                # Assign a unique color to each category
                color = colors[j % len(colors)]

                # Create scatterplot and add to list
                sc = ax.scatter(x, y, s=3, c=color, alpha=norm_weights, label=cat.name)
                self.scatterplots.append(sc)

            ax.legend()

        # After creating scatter plots, add empty lines for mean traces
        for i, agent in enumerate(self.agents):
            ax = self.axs[i]
            for j, cat in enumerate(agent.categories):
                line, = ax.plot([], [], '-', color='black', alpha=1)
                self.mean_lines.append(line)
        
        # Add empty text annotations for each category
        for i, agent in enumerate(self.agents):
            ax = self.axs[i]
            for cat in agent.categories:
                annotation = ax.text(0, 0, '', fontsize=8)
                self.text_annotations.append(annotation)
                
        return self.scatterplots