from matplotlib.colors import LinearSegmentedColormap

# https://en.wikipedia.org/wiki/Stellar_classification
stops = [
    (0.00, "#FF856C"),   #   red               #   < 2,300 K
    (0.31, "#FFB56C"),   #   orange-red        #   2,300–3,900 K
    (0.76, "#FFDAB5"),   #   yellow-orange     #   3,900–5,300 K
    (0.84, "#FFEDE3"),   #   yell w-white      #   5,300–6,000 K
    (0.92, "#F9F5FF"),   #   white             #   6,000–7,300 K
    (0.96, "#D5E0FF"),   #   blue-white        #   7,300–10,000 K
    (0.98, "#A2C0FF"),   #   deep-blue-white   #   10,000–33,000 K
    (1.00, "#92B5FF"),   #   blue              #   ≥ 33,000 K
]

cmap = LinearSegmentedColormap.from_list("stellar", stops, N=256)