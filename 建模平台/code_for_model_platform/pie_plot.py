import matplotlib.pyplot as plt

def pie_drawing(vals, labels,title):
    '''
    ax.pie(vals, labels=labels, colors=colors,
      autopct='%1.1f%%', shadow=True, startangle=90,radius=1.2)
    '''
    fig, ax = plt.subplots()  # 创建子图

    ax.pie(vals, radius=1, autopct='%1.1f%%', pctdistance=0.75)
    ax.pie([1], radius=0.6, colors='w')
    ax.set(aspect="equal", title=title)
    # plt.legend()
    plt.legend(labels, bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.)
    plt.show()
    print('')