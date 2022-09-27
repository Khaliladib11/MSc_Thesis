import json
import os.path

import matplotlib.pyplot as plt

x = [256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536]

study = {
    'yolov7': {
        'speed': [1.6, 2.6, 3.4, 4.9, 6.3, 8.6],
        'map': [0.19, 0.29, 0.34, 0.36, 0.375, 0.376]
    },
    'yolov7x': {
        'speed': [2.1, 3.4, 5.2, 7.2, 10.2, 14],
        'map': [0.198, 0.3, 0.353, 0.37, 0.38, 0.38]
    },
    'yolov5s': {
        'speed': [1.1, 1.6, 2.2, 2.7, 3.7, 4.9],
        'map': [0.1, 0.17, 0.23, 0.27, 0.3, 0.314]
    },
    'yolov5l': {
        'speed': [2.3, 4, 6.2, 8.7, 12.1, 16.9],
        'map': [0.156, 0.25, 0.316, 0.35, 0.371, 0.378]
    },
    'yolov5x': {
        'speed': [3.5, 6.7, 10.6, 15.2, 20.6, 29.4],
        'map': [0.165, 0.262, 0.328, 0.361, 0.381, 0.387]
    },
    'yolov6': {
        'speed': [0, 0, 0, 3.54, 4.64, 6.26],
        'map': [0, 0, 0, 0.27, 0.28, 0.28]
    }
}

test_order = [
    'b71d7574-e6bc43e9.jpg',
    'c4eb5462-657f5921.jpg',
    'b6818d65-6cfd37c8.jpg',
    'b23493b1-3200de1c.jpg',
    'bbf48d08-f32427fa.jpg',
    'c7648772-cbd18adf.jpg',
    'b5465c6e-b2fb645f.jpg',
    'c4e2eba0-02c66276.jpg',
    'c3593016-1adba20d.jpg',
    'c807cb19-7e09cb11.jpg',
    'bb2c5719-38a69465.jpg',
    'c50faaad-a8463d3d.jpg',
    'c845b617-34b9e98a.jpg',
    'bba686c2-2e7b039d.jpg',
    'b57dee9d-e5bd3142.jpg',
    'bf1af4ce-dcf62242.jpg',
]

#7204a601-5bed616c.jpg
#04629c39-1d564d8a.jpg

results_rtx5000 = {
    'yolov7': {
        'speed': 12.8,
        'map': 0.36,
    },
    'yolov7x': {
        'speed': 14.3,
        'map': 0.37,
    },
    'yolov5s': {
        'speed': 11,
        'map': 0.27,
    },
    'yolov5l': {
        'speed': 19,
        'map': 0.35,
    },
    'yolov5x': {
        'speed': 24,
        'map': 0.361,
    },
    'yolov6': {
        'speed': 8,
        'map': 0.28,
    },
    'fasterrcnn': {
        'speed': 86,
        'map': 0.31,
    },
}

results_a500 = {
    'yolov7': {
        'speed': 9.3,
        'map': 0.36
    },
    'yolov7x': {
        'speed': 10.5,
        'map': 0.37
    },
    'yolov5s': {
        'speed': 11.5,
        'map': 0.27
    },
    'yolov5l': {
        'speed': 15.6,
        'map': 0.31
    },
    'yolov5x': {
        'speed': 18,
        'map': 0.361
    },
    'yolov6': {
        'speed': 7.4,
        'map': 0.27
    },
    'fasterrcnn': {
        'speed': 26.19,
        'map': 0.31
    }
}

colors = ['orange', 'blue', 'green', 'gray', 'olive', 'brown', 'red']


def plot_scatter(results, title, save_path, xlabel='speed', ylabel='COCO AP'):
    plt.figure(figsize=(10, 7))
    for i, algo in enumerate(results):
        x = results[algo]['speed']
        y = results[algo]['map']
        plt.scatter(x, y, 200, label=algo, cmap=plt.cm.coolwarm)

    plt.legend()
    plt.xlabel(xlabel, size=20)
    plt.ylabel(ylabel, size=20)
    plt.title(title, size=20)
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def plot_study(results_dic, save_path):
    plt.figure(figsize=(10, 7))
    for alog in results_dic:
        speed = results_dic[alog]['speed']
        map = [item * 100 for item in results_dic[alog]['map']]
        plt.plot(speed, map, '.-', label=alog, linewidth=2, markersize=15)
    plt.legend()
    plt.xlabel('speed (img/ms)', size=20)
    plt.ylabel('COCO AP', size=20)
    plt.title("Inference time in ms vs COCO AP for different image sizes", size=20)
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def plot_barcharts(models, key, xlabel, ylabel, title, save_path):
    fig, ax = plt.subplots()

    x = []
    y = []
    for model in models:
        x.append(model)
        y.append(models[model][key] * 100)

    # Save the chart so we can loop through the bars below.
    bars = ax.bar(x, y, color=colors)

    # Axis formatting.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    # Add text annotations to the top of the bars.
    bar_color = bars[0].get_facecolor()
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            round(bar.get_height(), 1),
            horizontalalignment='center',
            color=colors[i],
            weight='bold'
        )

    # Add labels and a title. Note the use of `labelpad` and `pad` to add some
    # extra space between the text and the tick labels.
    ax.set_xlabel(xlabel, labelpad=15, color='#333333')
    ax.set_ylabel(ylabel, labelpad=15, color='#333333')
    ax.set_title(title, pad=15, color='#333333', weight='bold')

    fig.tight_layout()
    plt.savefig(save_path)


path_to_save = 'doc/images'
plot_scatter(results_a500,
             title='Inference time in ms vs COCO AP on A500 GPU',
             save_path=os.path.join(path_to_save, 'map_speed_a500.jpg'))
plot_scatter(results_rtx5000,
             title='Inference time in ms vs COCO AP on RTX5000 GPU',
             save_path=os.path.join(path_to_save, 'map_speed_rtx5000.jpg'))
plot_study(study, save_path=os.path.join(path_to_save, 'study.jpg'))
