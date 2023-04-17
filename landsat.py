from categorization import *

MOD = 0.02

def landsat_input(fileno, df):
    satimg = Image.open(f"Satellite Bands\LC09_CU_022009_20220721_20220726_02_SR_B{fileno}.TIF")
    satarr = np.array(satimg)
    for x in range(len(satarr[0])):
        if satarr[0][x] != 0:
            imgedge = x
            break

    satarr = np.delete(satarr, np.s_[:imgedge], 1)
    height = len(satarr)
    width = len(satarr[0])
    n = 0
    m = 0
    bandarr = []
    while(n < height):
        while(m < width):
            newarr = satarr[n : n + math.ceil(height * MOD), m : m + math.ceil(width * MOD)]
            arravg = rgb_average(newarr)
            arravg = (arravg - 8000) * (256 / 3000)
            bandarr.append(arravg)
            m += math.ceil(width * MOD)
        m = 0
        n += int(height * MOD)
    match fileno:
        case 2: 
            df['red'] = bandarr
        case 3:
            df['green'] = bandarr
        case 4:
            df['blue'] = bandarr
        case 5:
            df['infrared'] = bandarr
    #plt.plot(bandarr)
    #plt.savefig(f'values-{fileno}.jpg')
    #plt.show()

def distributions(cat, means, sds):
    rrange = np.arange(means[0] - 3 * sds[0], means[0] + 3 * sds[0], 0.01)
    grange = np.arange(means[1] - 3 * sds[1], means[1] + 3 * sds[1], 0.01)
    brange = np.arange(means[2] - 3 * sds[2], means[2] + 3 * sds[2], 0.01)
    rdist = sci.norm.pdf(rrange, means[0], sds[0])
    gdist = sci.norm.pdf(grange, means[1], sds[1])
    bdist = sci.norm.pdf(brange, means[2], sds[2])
    
    """figure, axis = plt.subplots(1, 3)
    figure.tight_layout()
    axis[0].plot(rrange, rdist, color = "red")
    axis[1].plot(grange, gdist, color= "green")
    axis[2].plot(brange, bdist, color = "blue")
    plt.savefig(f"{cat}_distributions.jpg")
    plt.show()"""

    return (rdist, gdist, bdist)
    

def cat_predictor(clusters, df):
    categories = [k for k in clusters.keys()]
    catmap = []
    chosen = ""

    for entry in range(df.shape[0]):
        for col in range(df.shape[1] - 1):
            probmax = 0
            rgbprobs = []
            rgbcats = []
            for cat in range(len(categories)):
                mean = clusters[categories[cat]][0]
                stdev = clusters[categories[cat]][1]
                zscore = (df.iat[entry, col] - mean[col]) / stdev[col]
                prob = abs(sci.norm.cdf(zscore + 0.01) - sci.norm.cdf(zscore - 0.01))
                if prob > probmax:
                    probmax = prob
                    chosen = categories[cat]
            rgbprobs.append(probmax)
            rgbcats.append(chosen)
        catmap.append(rgbcats[rgbprobs.index(max(rgbprobs))])
    catmap = np.reshape(catmap, (int(1 / MOD), int(1 / MOD)))
    #catmap = [[(catmap[j][i - 2], catmap[j][i - 1], catmap[j][i]) for i in range(2, len(catmap[j]), 3)] for j in range(len(catmap))] 
    
    return catmap

def plot_map(catmap):
    dct = {'agricultural': 0, 'forest': 1, 'buildings': 2, 'river': 3, 'chaparral': 4}
    m = [[dct[i] for i in j] for j in catmap]
    colors = "lime green gray blue yellow".split()
    cmap = clr.ListedColormap(colors, name = "colors", N=None)

    plt.imshow(m, cmap = cmap)
    plt.savefig('catmap.jpg')
    plt.show()

def main():
    groupdata, dataframe = datagen()
    #test_train(groupdata)
    clusterdata = clustering(dataframe)
    #datavis(dataframe)

    testdf = pd.DataFrame()
    for n in range(2, 6):
        landsat_input(n, testdf)
    #print(testdf)
    mappedcategories = cat_predictor(clusterdata, testdf)
    plot_map(mappedcategories)

if __name__ == "__main__":
    main()