from matplotlib import pyplot as plt
import numpy as np

for plt_nsites in [4, 6, 8]:
    for plt_nqubits0 in range(1, plt_nsites):

        with open('data.tsv') as f:
            lines = f.readlines()

            x1 = []
            x2 = []
            x3 = []
            x4 = []
            
            y1 = []
            y2 = []
            y3 = []
            y4 = []

            for line in lines:

                min_loss = float(line.split()[1])
                nsites = int(line.split()[2])
                nqubits0 = int(line.split()[3])
                h = float(line.split()[4])
                nlayers = int(line.split()[5])
                entropy = float(line.split()[6])

                if nsites == plt_nsites:
                    if nqubits0 == plt_nqubits0:

                        if nlayers == 1:
                            x1.append(h)
                            y1.append(min_loss)

                        if nlayers == 2:
                            x2.append(h)
                            y2.append(min_loss)

                        if nlayers == 3:
                            x3.append(h)
                            y3.append(min_loss)

                        if nlayers == 4:
                            x4.append(h)
                            y4.append(min_loss)


            if x1 or x2 or x3 or x4:
                fig, axs = plt.subplots(ncols=2, nrows=2, sharex = True)
                fig.suptitle("4 sites, 1 qubit0")

                v_labelsize = 7
                v_fontsize = 8

                ax = axs[0,0]

                ax.scatter(x1, y1, s=5, marker='o', label="1 layer")
                ax.set_ylabel("min_loss")
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                ax.legend(fontsize=v_fontsize)
                ax.tick_params(labelsize=v_labelsize)
                ax.yaxis.get_offset_text().set_fontsize(v_fontsize)


                ax = axs[0,1]

                ax.scatter(x2, y2, s=5, marker="o", label="2 layers")
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                ax.legend(fontsize=v_fontsize)
                ax.tick_params(labelsize=v_labelsize)
                ax.yaxis.get_offset_text().set_fontsize(v_fontsize)


                ax = axs[1,0]

                ax.scatter(x3, y3, s=5, marker="o", label="3 layers")
                ax.set_xlabel("h")
                ax.set_ylabel("min_loss")
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                ax.legend(fontsize=v_fontsize)
                ax.tick_params(labelsize=v_labelsize)
                ax.yaxis.get_offset_text().set_fontsize(v_fontsize)


                ax = axs[1,1]

                ax.scatter(x4, y4, s=5, marker="o", label="4 layers")
                ax.set_xlabel("h")
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                ax.legend(fontsize=v_fontsize)
                ax.tick_params(labelsize=v_labelsize)
                ax.yaxis.get_offset_text().set_fontsize(v_fontsize)

                figure_name = "nsites = " + str(plt_nsites) + ", nqubits0 = " + str(plt_nqubits0) + ".png"

                plt.savefig(figure_name, dpi=200)