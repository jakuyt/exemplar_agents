import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ExemplarTheoryPy as et

def create_agent(name, vowels, cat_size):
    agent = et.Agent(name)
    for vowel in vowels:
        cat = et.Category(vowel, mean=et.vowel_data[vowel], num_exemplars=cat_size)
        agent.add_category(cat)
    return agent

def main(config):
    agents = [
        create_agent('Alice', ['i', 'e', 'a', 'o', 'u'], config['cat_size']),
        create_agent('Bob', ['i', 'e', 'a', 'o', 'u'], config['cat_size'])
    ]

    fig, axs = plt.subplots(1, len(agents))
    model = et.Environment(agents, axs)

    ani = animation.FuncAnimation(
        fig,
        init_func=model.initialize_fig,
        func=model.update_exemplars,
        frames=config['n_time_steps'],
        blit=False,
        interval=config['frame_interval'],
        cache_frame_data=False,
        repeat=False,
        fargs=(
            config['decay_rate'],
            config['weight_threshold'],
            config['environmental_bias'],
            config['noise_std']
        )
    )

    plt.rcParams['animation.html'] = 'html5'
    fig.set_dpi(80)
    plt.gcf().set_size_inches(12, 6)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    main(config)