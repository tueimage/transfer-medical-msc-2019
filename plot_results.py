import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# eerst resultaten in arrays zetten
transfer_results = pd.read_csv('results/transfer_results.csv')
SVM_results = pd.read_csv('results/SVM_results.csv')
FT_results = pd.read_csv('results/FT_results.csv')
shift_results = pd.read_csv('results/shift_results.csv')

print(SVM_results)

transfer_names = transfer_results.source_dataset
transfer_AUCs = transfer_results.AUC
transfer_skl_AUCs = transfer_results.skl_AUC
transfer_ACCs = transfer_results.ACC

# print(transfer_AUCs, transfer_skl_AUCs, transfer_ACCs)
x_pct = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
baseline = [0.877, 0.877, 0.877, 0.877, 0.877, 0.877, 0.877, 0.877, 0.877, 0.877]
# baseline = [0.797, 0.797, 0.797, 0.797, 0.797, 0.797, 0.797, 0.797, 0.797, 0.797]
# baseline = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# baseline = [0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]


# transfer results
transfer_skAUC_image_rot = list(x for x in transfer_results.skl_AUC[:10])
transfer_skAUC_image_translation = list(x for x in transfer_results.skl_AUC[10:20])
transfer_skAUC_image_zoom = list(x for x in transfer_results.skl_AUC[20:30])
transfer_skAUC_add_noise_gaussian = list(x for x in transfer_results.skl_AUC[30:40])
transfer_skAUC_add_noise_poisson = list(x for x in transfer_results.skl_AUC[40:50])
transfer_skAUC_add_noise_salt_and_pepper = list(x for x in transfer_results.skl_AUC[50:60])
transfer_skAUC_add_noise_speckle = list(x for x in transfer_results.skl_AUC[60:70])
transfer_skAUC_imbalance_classes = list(x for x in transfer_results.skl_AUC[70:80])

transfer_ACC_image_rot = list(x for x in transfer_results.ACC[:10])
transfer_ACC_image_translation = list(x for x in transfer_results.ACC[10:20])
transfer_ACC_image_zoom = list(x for x in transfer_results.ACC[20:30])
transfer_ACC_add_noise_gaussian = list(x for x in transfer_results.ACC[30:40])
transfer_ACC_add_noise_poisson = list(x for x in transfer_results.ACC[40:50])
transfer_ACC_add_noise_salt_and_pepper = list(x for x in transfer_results.ACC[50:60])
transfer_ACC_add_noise_speckle = list(x for x in transfer_results.ACC[60:70])
transfer_ACC_imbalance_classes = list(x for x in transfer_results.ACC[70:80])

# SVM results
SVM_skAUC_image_rot = list(x for x in SVM_results.skl_AUC[:10])
SVM_skAUC_image_translation = list(x for x in SVM_results.skl_AUC[10:20])
SVM_skAUC_image_zoom = list(x for x in SVM_results.skl_AUC[20:30])
SVM_skAUC_add_noise_gaussian = list(x for x in SVM_results.skl_AUC[30:40])
SVM_skAUC_add_noise_poisson = list(x for x in SVM_results.skl_AUC[40:50])
SVM_skAUC_add_noise_salt_and_pepper = list(x for x in SVM_results.skl_AUC[50:60])
SVM_skAUC_add_noise_speckle = list(x for x in SVM_results.skl_AUC[60:70])
SVM_skAUC_imbalance_classes = list(x for x in SVM_results.skl_AUC[70:80])

# print(SVM_skAUC_image_rot)
# print(SVM_skAUC_image_translation)
# print(SVM_skAUC_image_zoom)
# print(SVM_skAUC_add_noise_gaussian)
# print(SVM_skAUC_add_noise_poisson)
# print(SVM_skAUC_add_noise_salt_and_pepper)
# print(SVM_skAUC_add_noise_speckle)


SVM_ACC_image_rot = list(x for x in SVM_results.ACC[:10])
SVM_ACC_image_translation = list(x for x in SVM_results.ACC[10:20])
SVM_ACC_image_zoom = list(x for x in SVM_results.ACC[20:30])
SVM_ACC_add_noise_gaussian = list(x for x in SVM_results.ACC[30:40])
SVM_ACC_add_noise_poisson = list(x for x in SVM_results.ACC[40:50])
SVM_ACC_add_noise_salt_and_pepper = list(x for x in SVM_results.ACC[50:60])
SVM_ACC_add_noise_speckle = list(x for x in SVM_results.ACC[60:70])
SVM_ACC_imbalance_classes = list(x for x in SVM_results.ACC[70:80])

# FT results (only classifier, no fine-tuning)
FTCLF_skAUC_image_rot = list(x for x in FT_results.skl_AUC[:10])
FTCLF_skAUC_image_translation = list(x for x in FT_results.skl_AUC[10:20])
FTCLF_skAUC_image_zoom = list(x for x in FT_results.skl_AUC[20:30])
FTCLF_skAUC_add_noise_gaussian = list(x for x in FT_results.skl_AUC[30:40])
FTCLF_skAUC_add_noise_poisson = list(x for x in FT_results.skl_AUC[40:50])
FTCLF_skAUC_add_noise_salt_and_pepper = list(x for x in FT_results.skl_AUC[50:60])
FTCLF_skAUC_add_noise_speckle = list(x for x in FT_results.skl_AUC[60:70])
FTCLF_skAUC_imbalance_classes = list(x for x in FT_results.skl_AUC[70:80])

FTCLF_ACC_image_rot = list(x for x in FT_results.ACC[:10])
FTCLF_ACC_image_translation = list(x for x in FT_results.ACC[10:20])
FTCLF_ACC_image_zoom = list(x for x in FT_results.ACC[20:30])
FTCLF_ACC_add_noise_gaussian = list(x for x in FT_results.ACC[30:40])
FTCLF_ACC_add_noise_poisson = list(x for x in FT_results.ACC[40:50])
FTCLF_ACC_add_noise_salt_and_pepper = list(x for x in FT_results.ACC[50:60])
FTCLF_ACC_add_noise_speckle = list(x for x in FT_results.ACC[60:70])
FTCLF_ACC_imbalance_classes = list(x for x in FT_results.ACC[70:80])

# FT results (with fine-tuning)
FT_skAUC_image_rot = list(x for x in FT_results.skl_AUC_ft[:10])
FT_skAUC_image_translation = list(x for x in FT_results.skl_AUC_ft[10:20])
FT_skAUC_image_zoom = list(x for x in FT_results.skl_AUC_ft[20:30])
FT_skAUC_add_noise_gaussian = list(x for x in FT_results.skl_AUC_ft[30:40])
FT_skAUC_add_noise_poisson = list(x for x in FT_results.skl_AUC_ft[40:50])
FT_skAUC_add_noise_salt_and_pepper = list(x for x in FT_results.skl_AUC_ft[50:60])
FT_skAUC_add_noise_speckle = list(x for x in FT_results.skl_AUC_ft[60:70])
FT_skAUC_imbalance_classes = list(x for x in FT_results.skl_AUC_ft[70:80])

FT_ACC_image_rot = list(x for x in FT_results.ACC_ft[:10])
FT_ACC_image_translation = list(x for x in FT_results.ACC_ft[10:20])
FT_ACC_image_zoom = list(x for x in FT_results.ACC_ft[20:30])
FT_ACC_add_noise_gaussian = list(x for x in FT_results.ACC_ft[30:40])
FT_ACC_add_noise_poisson = list(x for x in FT_results.ACC_ft[40:50])
FT_ACC_add_noise_salt_and_pepper = list(x for x in FT_results.ACC_ft[50:60])
FT_ACC_add_noise_speckle = list(x for x in FT_results.ACC_ft[60:70])
FT_ACC_imbalance_classes = list(x for x in FT_results.ACC_ft[70:80])



# get shift results
shift_AUC_image_rot = list(round(abs(x-0.5),3) for x in shift_results.skl_AUC[:10])
shift_AUC_image_translation = list(round(abs(x-0.5),3) for x in shift_results.skl_AUC[10:20])
shift_AUC_image_zoom = list(round(abs(x-0.5),3) for x in shift_results.skl_AUC[20:30])
shift_AUC_add_noise_gaussian = list(round(abs(x-0.5),3) for x in shift_results.skl_AUC[30:40])
shift_AUC_add_noise_poisson = list(round(abs(x-0.5),3) for x in shift_results.skl_AUC[40:50])
shift_AUC_add_noise_salt_and_pepper = list(round(abs(x-0.5),3) for x in shift_results.skl_AUC[50:60])
shift_AUC_add_noise_speckle = list(round(abs(x-0.5),3) for x in shift_results.skl_AUC[60:70])
shift_AUC_imbalance_classes = list(round(abs(x-0.5),3) for x in shift_results.skl_AUC[70:80])

shift_p_image_rot = list(x for x in shift_results.p_val_binom[:10])
shift_p_image_translation = list(x for x in shift_results.p_val_binom[10:20])
shift_p_image_zoom = list(x for x in shift_results.p_val_binom[20:30])
shift_p_add_noise_gaussian = list(x for x in shift_results.p_val_binom[30:40])
shift_p_add_noise_poisson = list(x for x in shift_results.p_val_binom[40:50])
shift_p_add_noise_salt_and_pepper = list(x for x in shift_results.p_val_binom[50:60])
shift_p_add_noise_speckle = list(x for x in shift_results.p_val_binom[60:70])
shift_p_imbalance_classes = list(x for x in shift_results.p_val_binom[70:80])





# create dataframe
df_SVM_AUC = pd.DataFrame({'fraction of images modified': x_pct})
df_SVM_AUC['rotation'] = SVM_skAUC_image_rot
df_SVM_AUC['translation'] = SVM_skAUC_image_translation
df_SVM_AUC['zoom'] = SVM_skAUC_image_zoom
df_SVM_AUC['gaussian noise'] = SVM_skAUC_add_noise_gaussian
df_SVM_AUC['poisson noise'] = SVM_skAUC_add_noise_poisson
df_SVM_AUC['salt and pepper noise'] = SVM_skAUC_add_noise_salt_and_pepper
df_SVM_AUC['speckle noise'] = SVM_skAUC_add_noise_speckle
df_SVM_AUC['class imbalance'] = SVM_skAUC_imbalance_classes
df_SVM_AUC['baseline'] = baseline

# df = pd.DataFrame.copy(df_SVM_AUC)

# change from wide to long format as sns only accepts dataframes with two columns (x and y)
df_SVM_AUC = pd.melt(df_SVM_AUC, ['fraction of images modified'])
df_SVM_AUC = df_SVM_AUC.rename(columns={'value': 'AUC'})


plt.figure()
sns.set(style="darkgrid")
# flatui = ['#A569BD', '#85C1E9', '#48C9B0', '#fc8d59', '#ef6548', '#d7301f', '#990000', '#273746']
sns.set_palette("Set3")
# sns.set_palette(flatui)
sns.lineplot(x='fraction of images modified', y='AUC', hue="variable", data=df_SVM_AUC, linewidth=2.5)
plt.show()







# create dataframe
df_FTCLF_AUC = pd.DataFrame({'fraction of images modified': x_pct})
df_FTCLF_AUC['rotation'] = FTCLF_skAUC_image_rot
df_FTCLF_AUC['translation'] = FTCLF_skAUC_image_translation
df_FTCLF_AUC['zoom'] = FTCLF_skAUC_image_zoom
df_FTCLF_AUC['gaussian noise'] = FTCLF_skAUC_add_noise_gaussian
df_FTCLF_AUC['poisson noise'] = FTCLF_skAUC_add_noise_poisson
df_FTCLF_AUC['salt and pepper noise'] = FTCLF_skAUC_add_noise_salt_and_pepper
df_FTCLF_AUC['speckle noise'] = FTCLF_skAUC_add_noise_speckle
df_FTCLF_AUC['class imbalance'] = FTCLF_skAUC_imbalance_classes
df_FTCLF_AUC['baseline'] = baseline

# df = pd.DataFrame.copy(df_FTCLF_AUC)

# change from wide to long format as sns only accepts dataframes with two columns (x and y)
df_FTCLF_AUC = pd.melt(df_FTCLF_AUC, ['fraction of images modified'])
df_FTCLF_AUC = df_FTCLF_AUC.rename(columns={'value': 'AUC'})


plt.figure()
sns.set(style="darkgrid")
# flatui = ['#A569BD', '#85C1E9', '#48C9B0', '#fc8d59', '#ef6548', '#d7301f', '#990000', '#273746']
sns.set_palette("Set3")
# sns.set_palette(flatui)
sns.lineplot(x='fraction of images modified', y='AUC', hue="variable", data=df_FTCLF_AUC, linewidth=2.5)
plt.show()







# create dataframe
df_FT_AUC = pd.DataFrame({'fraction of images modified': x_pct})
df_FT_AUC['rotation'] = FT_skAUC_image_rot
df_FT_AUC['translation'] = FT_skAUC_image_translation
df_FT_AUC['zoom'] = FT_skAUC_image_zoom
df_FT_AUC['gaussian noise'] = FT_skAUC_add_noise_gaussian
df_FT_AUC['poisson noise'] = FT_skAUC_add_noise_poisson
df_FT_AUC['salt and pepper noise'] = FT_skAUC_add_noise_salt_and_pepper
df_FT_AUC['speckle noise'] = FT_skAUC_add_noise_speckle
df_FT_AUC['class imbalance'] = FT_skAUC_imbalance_classes
df_FT_AUC['baseline'] = baseline

# df = pd.DataFrame.copy(df_FT_AUC)

# change from wide to long format as sns only accepts dataframes with two columns (x and y)
df_FT_AUC = pd.melt(df_FT_AUC, ['fraction of images modified'])
df_FT_AUC = df_FT_AUC.rename(columns={'value': 'AUC'})


plt.figure()
sns.set(style="darkgrid")
# flatui = ['#A569BD', '#85C1E9', '#48C9B0', '#fc8d59', '#ef6548', '#d7301f', '#990000', '#273746']
sns.set_palette("Set3")
# sns.set_palette(flatui)
sns.lineplot(x='fraction of images modified', y='AUC', hue="variable", data=df_FT_AUC, linewidth=2.5)
plt.show()

######################### ACC

# create dataframe
df_SVM_ACC = pd.DataFrame({'fraction of images modified': x_pct})
df_SVM_ACC['rotation'] = SVM_ACC_image_rot
df_SVM_ACC['translation'] = SVM_ACC_image_translation
df_SVM_ACC['zoom'] = SVM_ACC_image_zoom
df_SVM_ACC['gaussian noise'] = SVM_ACC_add_noise_gaussian
df_SVM_ACC['poisson noise'] = SVM_ACC_add_noise_poisson
df_SVM_ACC['salt and pepper noise'] = SVM_ACC_add_noise_salt_and_pepper
df_SVM_ACC['speckle noise'] = SVM_ACC_add_noise_speckle
df_SVM_ACC['class imbalance'] = SVM_ACC_imbalance_classes
df_SVM_ACC['baseline'] = baseline

# df = pd.DataFrame.copy(df_SVM_ACC)

# change from wide to long format as sns only accepts dataframes with two columns (x and y)
df_SVM_ACC = pd.melt(df_SVM_ACC, ['fraction of images modified'])
df_SVM_ACC = df_SVM_ACC.rename(columns={'value': 'ACC'})


plt.figure()
sns.set(style="darkgrid")
# flatui = ['#A569BD', '#85C1E9', '#48C9B0', '#fc8d59', '#ef6548', '#d7301f', '#990000', '#273746']
sns.set_palette("Set3")
# sns.set_palette(flatui)
sns.lineplot(x='fraction of images modified', y='ACC', hue="variable", data=df_SVM_ACC, linewidth=2.5)
plt.show()




# create dataframe
df_FTCLF_ACC = pd.DataFrame({'fraction of images modified': x_pct})
df_FTCLF_ACC['rotation'] = FTCLF_ACC_image_rot
df_FTCLF_ACC['translation'] = FTCLF_ACC_image_translation
df_FTCLF_ACC['zoom'] = FTCLF_ACC_image_zoom
df_FTCLF_ACC['gaussian noise'] = FTCLF_ACC_add_noise_gaussian
df_FTCLF_ACC['poisson noise'] = FTCLF_ACC_add_noise_poisson
df_FTCLF_ACC['salt and pepper noise'] = FTCLF_ACC_add_noise_salt_and_pepper
df_FTCLF_ACC['speckle noise'] = FTCLF_ACC_add_noise_speckle
df_FTCLF_ACC['class imbalance'] = FTCLF_ACC_imbalance_classes
df_FTCLF_ACC['baseline'] = baseline

# df = pd.DataFrame.copy(df_FTCLF_ACC)

# change from wide to long format as sns only accepts dataframes with two columns (x and y)
df_FTCLF_ACC = pd.melt(df_FTCLF_ACC, ['fraction of images modified'])
df_FTCLF_ACC = df_FTCLF_ACC.rename(columns={'value': 'ACC'})


plt.figure()
sns.set(style="darkgrid")
# flatui = ['#A569BD', '#85C1E9', '#48C9B0', '#fc8d59', '#ef6548', '#d7301f', '#990000', '#273746']
sns.set_palette("Set3")
# sns.set_palette(flatui)
sns.lineplot(x='fraction of images modified', y='ACC', hue="variable", data=df_FTCLF_ACC, linewidth=2.5)
plt.show()





# create dataframe
df_FT_ACC = pd.DataFrame({'fraction of images modified': x_pct})
df_FT_ACC['rotation'] = FT_ACC_image_rot
df_FT_ACC['translation'] = FT_ACC_image_translation
df_FT_ACC['zoom'] = FT_ACC_image_zoom
df_FT_ACC['gaussian noise'] = FT_ACC_add_noise_gaussian
df_FT_ACC['poisson noise'] = FT_ACC_add_noise_poisson
df_FT_ACC['salt and pepper noise'] = FT_ACC_add_noise_salt_and_pepper
df_FT_ACC['speckle noise'] = FT_ACC_add_noise_speckle
df_FT_ACC['class imbalance'] = FT_ACC_imbalance_classes
df_FT_ACC['baseline'] = baseline

# df = pd.DataFrame.copy(df_FT_ACC)

# change from wide to long format as sns only accepts dataframes with two columns (x and y)
df_FT_ACC = pd.melt(df_FT_ACC, ['fraction of images modified'])
df_FT_ACC = df_FT_ACC.rename(columns={'value': 'ACC'})


plt.figure()
sns.set(style="darkgrid")
# flatui = ['#A569BD', '#85C1E9', '#48C9B0', '#fc8d59', '#ef6548', '#d7301f', '#990000', '#273746']
sns.set_palette("Set3")
# sns.set_palette(flatui)
sns.lineplot(x='fraction of images modified', y='ACC', hue="variable", data=df_FT_ACC, linewidth=2.5)
plt.show()





# create dataframe
df_shift_AUC = pd.DataFrame({'fraction of images modified': x_pct})
df_shift_AUC['rotation'] = shift_AUC_image_rot
df_shift_AUC['translation'] = shift_AUC_image_translation
df_shift_AUC['zoom'] = shift_AUC_image_zoom
df_shift_AUC['gaussian noise'] = shift_AUC_add_noise_gaussian
df_shift_AUC['poisson noise'] = shift_AUC_add_noise_poisson
df_shift_AUC['salt and pepper noise'] = shift_AUC_add_noise_salt_and_pepper
df_shift_AUC['speckle noise'] = shift_AUC_add_noise_speckle
df_shift_AUC['class imbalance'] = shift_AUC_imbalance_classes
df_shift_AUC['baseline (random)'] = baseline

df = pd.DataFrame.copy(df_shift_AUC)

# change from wide to long format as sns only accepts dataframes with two columns (x and y)
df_shift_AUC = pd.melt(df_shift_AUC, ['fraction of images modified'])
df_shift_AUC = df_shift_AUC.rename(columns={'value': 'ACC'})


plt.figure()
sns.set(style="darkgrid")
# flatui = ['#A569BD', '#85C1E9', '#48C9B0', '#fc8d59', '#ef6548', '#d7301f', '#990000', '#273746']
sns.set_palette("Set3")
# sns.set_palette(flatui)
sns.lineplot(x='fraction of images modified', y='ACC', hue="variable", data=df_shift_AUC, linewidth=2.5)
plt.show()


# create dataframe
df_shift_p = pd.DataFrame({'fraction of images modified': x_pct})
df_shift_p['rotation'] = shift_p_image_rot
df_shift_p['translation'] = shift_p_image_translation
df_shift_p['zoom'] = shift_p_image_zoom
df_shift_p['gaussian noise'] = shift_p_add_noise_gaussian
df_shift_p['poisson noise'] = shift_p_add_noise_poisson
df_shift_p['salt and pepper noise'] = shift_p_add_noise_salt_and_pepper
df_shift_p['speckle noise'] = shift_p_add_noise_speckle
df_shift_p['class imbalance'] = shift_p_imbalance_classes
df_shift_p['baseline (significance level)'] = baseline

# df = pd.DataFrame.copy(df_shift_p)

# change from wide to long format as sns only accepts dataframes with two columns (x and y)
df_shift_p = pd.melt(df_shift_p, ['fraction of images modified'])
df_shift_p = df_shift_p.rename(columns={'value': 'ACC'})


plt.figure()
sns.set(style="darkgrid")
# flatui = ['#A569BD', '#85C1E9', '#48C9B0', '#fc8d59', '#ef6548', '#d7301f', '#990000', '#273746']
sns.set_palette("Set3")
# sns.set_palette(flatui)
sns.lineplot(x='fraction of images modified', y='ACC', hue="variable", data=df_shift_p, linewidth=2.5)
plt.show()




#
# # create dataframe
# df_transfer_AUC = pd.DataFrame({'fraction of images modified': x_pct})
# df_transfer_AUC['rotation'] = transfer_skAUC_image_rot
# df_transfer_AUC['translation'] = transfer_skAUC_image_translation
# df_transfer_AUC['zoom'] = transfer_skAUC_image_zoom
# df_transfer_AUC['gaussian noise'] = transfer_skAUC_add_noise_gaussian
# df_transfer_AUC['poisson noise'] = transfer_skAUC_add_noise_poisson
# df_transfer_AUC['salt and pepper noise'] = transfer_skAUC_add_noise_salt_and_pepper
# df_transfer_AUC['speckle noise'] = transfer_skAUC_add_noise_speckle
# df_transfer_AUC['baseline'] = baseline
#
# # df = pd.DataFrame.copy(df_transfer_AUC)
#
# # change from wide to long format as sns only accepts dataframes with two columns (x and y)
# df_transfer_AUC = pd.melt(df_transfer_AUC, ['fraction of images modified'])
# df_transfer_AUC = df_transfer_AUC.rename(columns={'value': 'AUC'})
#
#
# plt.figure()
# sns.set(style="darkgrid")
# flatui = ['#A569BD', '#85C1E9', '#48C9B0', '#fc8d59', '#ef6548', '#d7301f', '#990000', '#273746']
# # sns.set_palette("deep")
# sns.set_palette(flatui)
# sns.lineplot(x='fraction of images modified', y='AUC', hue="variable", data=df_transfer_AUC, linewidth=2.5)
# plt.show()










################ subplotjes

plt.figure()
# Initialize the figure
plt.style.use('seaborn-darkgrid')

# create a color palette
palette = plt.get_cmap('Set3')

# multiple line plot
num=0
for column in df.drop('fraction of images modified', axis=1):
    num+=1

    if num==5:
        plt.ylabel("$\Delta$AUC")
    if num==9:
        plt.xlabel("Fraction of images modified")

    # Find the right spot on the plot
    plt.subplot(3,3, num)

    # plot every groups, but discreet
    for v in df.drop('fraction of images modified', axis=1):
        plt.plot(df['fraction of images modified'], df[v], marker='', color='grey', linewidth=0.6, alpha=0.3)

    # Plot the lineplot
    plt.plot(df['fraction of images modified'], df[column], marker='', color=palette(num-1), linewidth=2.5, alpha=0.9, label=column)
    plt.plot(df['fraction of images modified'], baseline, '--', marker='', color='black', linewidth=1.5, alpha=0.6)

    # Same limits for everybody!
    plt.xlim(0.1,1.0)
    # plt.ylim(0.84,0.90)
    # plt.ylim(0.70,0.83)
    plt.ylim(0, 0.5)

    # Not ticks everywhere
    if num in range(7) :
        plt.tick_params(labelbottom='off')
    if num not in [1,4,7] :
        plt.tick_params(labelleft='off')

    # Add title
    plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num-1) )

# # create big subplot and hide frame, just for common axis labels
# plt.subplot(111, frameon=False)
# # hide tick and tick label of the big axes
# plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
# plt.grid(False)
# plt.xlabel("fraction of images modified")
# plt.ylabel("AUC")

# general title
# plt.suptitle("SVM classifier on features extracted with pre-trained model\n trained on varying percentage of modified images", fontsize=13, fontweight=0, color='black', style='italic', y=0.97)
# plt.suptitle("Fully Connected (FC) layers trained on top of pre-trained model\n trained on varying percentage of modified images", fontsize=13, fontweight=0, color='black', style='italic', y=0.97)
# plt.suptitle("Fine-tuning of pre-trained model\n trained on varying percentage of modified images", fontsize=13, fontweight=0, color='black', style='italic', y=0.97)
plt.suptitle("Difference in AUC from AUC=0.5 for shift detection classifier", fontsize=13, fontweight=0, color='black', style='italic', y=0.97)
# plt.suptitle("p-value of binomial test after shift detection classifier", fontsize=13, fontweight=0, color='black', style='italic', y=0.97)


# Axis title
# plt.text(0.5, 0.02, 'Time', ha='center', va='center')

plt.show()
