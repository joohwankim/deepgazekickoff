import pickle, pdb, os, glob, cv2
import matplotlib.pyplot as plt
import numpy as np

# convenience function for more readable code
def repeatAndJitter(x_val,x_len,jitter_magnitude): # for a given y vector, create an x vector that repeats the same value with random jitter.
    x = list()
    x.append(x_val)
    x = x * x_len
    x = x + (np.random.rand(len(x)) - 0.5) * jitter_magnitude
    return x

def loadPickledData(datapath):
    fp = open(datapath,'rb')
    data = pickle.load(fp)
    return data

def extendAnalysisOnSyntheticImage(model_name):
    def getSampleIntensity(img,coord):
        intensitySampleSize = 10
        intensityMatrix = img[
            int(coord[1] - intensitySampleSize/2):int(coord[1] + intensitySampleSize/2),
            int(coord[0] - intensitySampleSize/2):int(coord[0] + intensitySampleSize/2)
        ]
        return intensityMatrix

    # The sample synthetic images are generated for the same geometrical settings (same gaze direction, eye openness, pupil dilation, etc). So region annotation was performed for only one image (frame0020.png, which is randomly chosen) and then this function collects intensity data from all remaining images.
    try:
        regionData = loadPickledData(os.path.join('SyntheticSamples',model_name,'region_annotation.p'))
    except:
        print('Region annotation data is missing. Create it by running examineIntensity.py on a synthetic image.')
        sys.exit()
    pathLists = list()
    pathLists.append(glob.glob(os.path.join('SyntheticSamples',model_name,'Iris','*.png')))
    pathLists.append(glob.glob(os.path.join('SyntheticSamples',model_name,'Sclera','*.png')))
    pathLists.append(glob.glob(os.path.join('SyntheticSamples',model_name,'Skin','*.png')))
    if os.path.exists(os.path.join('SyntheticSamples',model_name,'Test')): # Test images exist?
        pathLists.append(glob.glob(os.path.join('SyntheticSamples',model_name,'Test','*.png')))
    for paths in pathLists:
        paths.sort()
        results = {
            'imgFileName':list(),
            'pupilCenter':list(),
            'pupilRadius':list(),
            'pupilSamples':{'coord':list(),'intensity':list()},
            'irisCenter':list(),
            'irisRadius':list(),
            'irisSamples':{'coord':list(),'intensity':list()},
            'scleraSamples':{'coord':list(),'intensity':list()},
            'skinSamples':{'coord':list(),'intensity':list()},
            'eyeOpening':list()
        }
        # process each image
        for fn in paths:
            # img = cv2.imread(fn)
            # pdb.set_trace()
            img = cv2.cvtColor(cv2.imread(fn),cv2.COLOR_RGB2GRAY)
            results['imgFileName'].append(fn)
            results['pupilCenter'].append(regionData['pupilCenter'][0])
            results['pupilRadius'].append(regionData['pupilRadius'][0])
            results['pupilSamples']['coord'].append(regionData['pupilSamples']['coord'][0])
            results['pupilSamples']['intensity'].append(getSampleIntensity(img, regionData['pupilSamples']['coord'][0][0]))
            results['irisCenter'].append(regionData['irisCenter'][0])
            results['irisRadius'].append(regionData['irisRadius'][0])
            results['irisSamples']['coord'].append(regionData['irisSamples']['coord'][0])
            results['irisSamples']['intensity'].append([getSampleIntensity(img, regionData['irisSamples']['coord'][0][0]),getSampleIntensity(img, regionData['irisSamples']['coord'][0][1])])
            results['scleraSamples']['coord'].append(regionData['scleraSamples']['coord'][0])
            results['scleraSamples']['intensity'].append([getSampleIntensity(img, regionData['scleraSamples']['coord'][0][0]),getSampleIntensity(img, regionData['scleraSamples']['coord'][0][1])])
            results['skinSamples']['coord'].append(regionData['skinSamples']['coord'][0])
            results['skinSamples']['intensity'].append([getSampleIntensity(img, regionData['skinSamples']['coord'][0][0]),getSampleIntensity(img, regionData['skinSamples']['coord'][0][1])])
            results['eyeOpening'].append(regionData['eyeOpening'][0])

        # save data to analysis.p file in the enclosing folder.
        analysisFileName = os.path.join(os.path.dirname(paths[0]),'analysis.p')
        fp = open(analysisFileName, 'wb')
        pickle.dump(results,fp,protocol = pickle.HIGHEST_PROTOCOL)

class RealDataScatterPlotHelper:
    def __init__(self, data_path, x_spacing = 2.0, jitter_magnitude = 0.5):
        # Load the analysis data on real images...
        # These are hard-coded for now. Could be passed as a variable if needed in future.
        self.data = loadPickledData(data_path)
        self.subjectList = ['AM','CF','EL','JHK','JSK','KA','MM','MS','NK','SB','WL'] 
        self.x_spacing = x_spacing
        self.jitter_magnitude = jitter_magnitude
        
    # NOTE: finding 'x' and 'y' repeats many operations. In principle we could reduce that to generate one matrix containing one column of x vector and one column of y vector. But then we need one more line of code when plotting it into scatter plot because matplotlib scatter plot does not receive a matrix containing both x and y vector (opposed to what matlab does, for example). I choose simpler code for drawing the plot than efficiency. Another workaround could be writing a wrapper for the scatter plot function. But that is going to be tedious and time-consuming.
    def get(self, x_y, data_description, min_max, stat_description):
        # x_y could be one of the followings:
        #     'x', 'y'
        # data_description could be one of the followings:
        #     'pupil', 'iris', 'sclera', 'skin'
        # min_max could be one of the followings:
        #     '' (only allowed for pupil), 'min', 'max'
        # stat_description could be one of the followings:
        #     'raw', 'mean'
        key = data_description + 'Samples'
        if min_max == 'min':
            mm_index = 0
        elif min_max == 'max':
            mm_index = 1
        else:
            mm_index = None
        output = list()
        for ii, fn in enumerate(self.data['imgFileName']):
            # TODO: skip if not looking straight ahead
            temp_str_list = str.split(fn,'_')
            gaze_x = float(temp_str_list[temp_str_list.index('X')+1])
            gaze_y = float(temp_str_list[temp_str_list.index('Y')+1])
            if np.linalg.norm((gaze_x,gaze_y),2) < 10.0:
                # Get both raw and mean, and x and y. I am not bothering to do only the necessary thing.
                s_index = self.subjectList.index(temp_str_list[0])
                if data_description == 'pupil':
                    raw_y = np.squeeze(np.reshape(self.data[key]['intensity'][ii][:],(1,-1)))
                else:
                    raw_y = np.squeeze(np.reshape(self.data[key]['intensity'][ii][mm_index][:],(1,-1)))
                mean_y = list()
                mean_y.append(np.mean(raw_y))
                raw_x = repeatAndJitter(s_index * self.x_spacing, len(raw_y), self.jitter_magnitude)
                mean_x = repeatAndJitter(s_index * self.x_spacing, len(mean_y), self.jitter_magnitude)
                # Finally, extend xx for what the caller of the function wanted.
                try:
                    exec('output.extend(%s_%s)'%(stat_description,x_y))
                except:
                    pdb.set_trace()
                    print('Stat not supported!!! Only "raw" or "mean" are supported.')
        return output

    def x(self, data_description, min_max, stat_description): # generate x vector of the desired data for scatter plot
        xx = self.get('x', data_description, min_max, stat_description)
        return xx

    def y(self, data_description, min_max, stat_description): # generate y vector of the desired data for scatter plot
        yy = self.get('y', data_description, min_max, stat_description)
        return yy

class SyntheticDataScatterPlotHelper:
    def __init__(self, data_path, x_spacing = 1.0, jitter_magnitude = 0.0):
        # Load the analysis data on synthetic images.
        # NOTE: Synthetic data is analyzed in a different way than the real data
        self.data = loadPickledData(data_path)
        self.x_spacing = x_spacing
        self.jitter_magnitude = jitter_magnitude
        
    # NOTE: finding 'x' and 'y' repeats many operations. In principle we could reduce that to generate one matrix containing one column of x vector and one column of y vector. But then we need one more line of code when plotting it into scatter plot because matplotlib scatter plot does not receive a matrix containing both x and y vector (opposed to what matlab does, for example). I choose simpler code for drawing the plot than efficiency. Another workaround could be writing a wrapper for the scatter plot function. But that is going to be tedious and time-consuming.
    def get(self, x_y, data_description, min_max, stat_description):
        # x_y could be one of the followings:
        #     'x', 'y'
        # data_description could be one of the followings:
        #     'pupil', 'iris', 'sclera', 'skin'
        # min_max could be one of the followings:
        #     '' (only allowed for pupil), 'min', 'max'
        # stat_description could be one of the followings:
        #     'raw', 'mean'
        key = data_description + 'Samples'
        if min_max == 'min':
            mm_index = 0
        elif min_max == 'max':
            mm_index = 1
        else:
            mm_index = None
        output = list()
        for ii, fn in enumerate(self.data['imgFileName']):
            # TODO: skip if not looking straight ahead
            temp_str_list = os.path.basename(fn).replace('frame','').replace('.png','')
            # Get both raw and mean, and x and y. I am not bothering to minimize the amount of operation to do only the necessary thing.
            s_index = None
            if temp_str_list.isdigit():
                s_index = int(temp_str_list)
            else:
                s_index = 0
            if data_description == 'pupil':
                raw_y = np.squeeze(np.reshape(self.data[key]['intensity'][ii][:],(1,-1)))
            else:
                raw_y = np.squeeze(np.reshape(self.data[key]['intensity'][ii][mm_index][:],(1,-1)))
            mean_y = list()
            mean_y.append(np.mean(raw_y))
            raw_x = repeatAndJitter(s_index * self.x_spacing, len(raw_y), self.jitter_magnitude)
            mean_x = repeatAndJitter(s_index * self.x_spacing, len(mean_y), self.jitter_magnitude)
            # Finally, extend xx for what the caller of the function wanted.
            try:
                exec('output.extend(%s_%s)'%(stat_description,x_y))
            except:
                print('Stat not supported!!! Only "raw" or "mean" are supported.')
        return output

    def x(self, data_description, min_max, stat_description): # generate x vector of the desired data for scatter plot
        xx = self.get('x', data_description, min_max, stat_description)
        return xx

    def y(self, data_description, min_max, stat_description): # generate y vector of the desired data for scatter plot
        yy = self.get('y', data_description, min_max, stat_description)
        return yy

model_name = 'male_05'
extendAnalysisOnSyntheticImage(model_name)

s_iris = SyntheticDataScatterPlotHelper(os.path.join('SyntheticSamples',model_name,'Iris','analysis.p'))
s_sclera = SyntheticDataScatterPlotHelper(os.path.join('SyntheticSamples',model_name,'Sclera','analysis.p'))
s_skin = SyntheticDataScatterPlotHelper(os.path.join('SyntheticSamples',model_name,'Skin','analysis.p'))
s_test = None
if os.path.exists(os.path.join('SyntheticSamples',model_name,'Test','analysis.p')):
    s_test = SyntheticDataScatterPlotHelper(os.path.join('SyntheticSamples',model_name,'Test','analysis.p'))
rr = RealDataScatterPlotHelper(os.path.join('RealSamples','analysis.p'), 10.0, 2.0)

# color setup
real_raw_color = 'r'
real_raw_marker = 'o'
real_raw_marker_size = 0.1
real_mean_color = 'm'
real_mean_marker = 's'
real_mean_marker_size = 7.0
synth_raw_color = 'b'
synth_raw_marker = 'o'
synth_raw_marker_size = 0.1
synth_mean_color = 'g'
synth_mean_marker = 's'
synth_mean_marker_size = 7.0
test_raw_color = 'c'
test_raw_marker = 'o'
test_raw_marker_size = 0.1
test_mean_color = 'k'
test_mean_marker = 's'
test_mean_marker_size = 7.0

# visualize intensity distribution per subject and multiplication factor
fig = plt.figure(1)
data_descriptions = ['pupil','iris','sclera','skin']
# below will be input to a function
ax = list()
for ii,desc in enumerate(data_descriptions):
    exec('ax.append(fig.add_subplot(2,2,%i))'%(ii+1))
    if desc == 'pupil':
        # plot results from real data
        ax[ii].scatter(rr.x(desc,'','raw'),rr.y(desc,'','raw'),c = real_raw_color, marker = real_raw_marker, s = real_raw_marker_size)
        ax[ii].scatter(rr.x(desc,'','mean'),rr.y(desc,'','mean'),c = real_mean_color, marker = real_mean_marker, s = real_mean_marker_size)
        # plot summary results from synthetic data
        ax[ii].scatter(s_skin.x(desc,'','raw'),s_skin.y(desc,'','raw'),c = synth_raw_color, marker = synth_raw_marker, s = synth_raw_marker_size)
        ax[ii].scatter(s_skin.x(desc,'','mean'),s_skin.y(desc,'','mean'),c = synth_mean_color, marker = synth_mean_marker, s = synth_mean_marker_size)
        # plot results from test image
        if s_test != None:
            ax[ii].scatter(s_test.x(desc,'','raw'),s_test.y(desc,'','raw'),c = test_raw_color, marker = test_raw_marker, s = test_raw_marker_size)
            ax[ii].scatter(s_test.x(desc,'','mean'),s_test.y(desc,'','mean'),c = test_mean_color, marker = test_mean_marker, s = test_mean_marker_size)
    else:
        # plot results from real data
        ax[ii].scatter(rr.x(desc,'min','raw'),rr.y(desc,'min','raw'),c = real_raw_color, marker = real_raw_marker, s = real_raw_marker_size)
        ax[ii].scatter(rr.x(desc,'min','mean'),rr.y(desc,'min','mean'),c = real_mean_color, marker = real_mean_marker, s = real_mean_marker_size)
        ax[ii].scatter(rr.x(desc,'max','raw'),rr.y(desc,'max','raw'),c = real_raw_color, marker = real_raw_marker, s = real_raw_marker_size)
        ax[ii].scatter(rr.x(desc,'max','mean'),rr.y(desc,'max','mean'),c = real_mean_color, marker = real_mean_marker, s = real_mean_marker_size)
        # plot summary results from synthetic data
        exec('s = s_%s'%desc)
        ax[ii].scatter(s.x(desc,'min','raw'),s.y(desc,'min','raw'),c = synth_raw_color, marker = synth_raw_marker, s = synth_raw_marker_size)
        ax[ii].scatter(s.x(desc,'min','mean'),s.y(desc,'min','mean'),c = synth_mean_color, marker = synth_mean_marker, s = synth_mean_marker_size)
        ax[ii].scatter(s.x(desc,'max','raw'),s.y(desc,'max','raw'),c = synth_raw_color, marker = synth_raw_marker, s = synth_raw_marker_size)
        ax[ii].scatter(s.x(desc,'max','mean'),s.y(desc,'max','mean'),c = synth_mean_color, marker = synth_mean_marker, s = synth_mean_marker_size)
        # plot results from test image
        if s_test != None:
            ax[ii].scatter(s_test.x(desc,'min','raw'),s_test.y(desc,'min','raw'),c = test_raw_color, marker = test_raw_marker, s = test_raw_marker_size)
            ax[ii].scatter(s_test.x(desc,'min','mean'),s_test.y(desc,'min','mean'),c = test_mean_color, marker = test_mean_marker, s = test_mean_marker_size)
            ax[ii].scatter(s_test.x(desc,'max','raw'),s_test.y(desc,'max','raw'),c = test_raw_color, marker = test_raw_marker, s = test_raw_marker_size)
            ax[ii].scatter(s_test.x(desc,'max','mean'),s_test.y(desc,'max','mean'),c = test_mean_color, marker = test_mean_marker, s = test_mean_marker_size)
    ax[ii].set_title(desc)

plt.show()
# realResults['pupilSamples']['intensity'][si]


# # understand gaze directions
# gaze_directions = list()
# subjects = list()
# for filename in results['imgFileName']:
#     s = filename.replace('.png', '').split('_')
#     subjects.append(s[0])
#     try:
#         if 'X' in s and 'Y' in s:
#             gaze_direction = (float(s[s.index('X') + 1]), float(s[s.index('Y') + 1]))
#         elif s[0] == 'From' and s[3] == 'To':
#             gaze_direction = (float(s[1]), float(s[2]), float(s[4]), float(s[5]))
#     except Exception as e:
#         print("Cannot parse filename '%s'." % os.path.basename(fn))
#         print(e)
#         exit()
#     gaze_directions.append(gaze_direction)
# # stat for central gaze. effect of pupil size change.
# irisRadii = [ir for ir, gd in zip(results['irisRadius'], gaze_directions) if np.linalg.norm(gd) < 10.0]
# pupilRadii = [pr for pr, gd in zip(results['pupilRadius'], gaze_directions) if np.linalg.norm(gd) < 10.0]
# pupilDisplacements = [(np.asarray(pc) - np.asarray(ic)).tolist() for pc, ic, gd in zip(results['pupilCenter'],results['irisCenter'],gaze_directions) if np.linalg.norm(gd) < 10.0]
# subjects = [s for s, gd in zip(subjects, gaze_directions) if np.linalg.norm(gd) < 10.0]
# fig = plt.figure(1)
# ax1 = fig.add_subplot(1,3,1)
# ax1_subjects = ['AM','CF','EL','JHK','JSK','KA','MM','MS','NK','SB','WL']
# ax1_colors = ['b','g','r','c','m','k','b','g','r','c','m']
# ax1_markers = ['o','o','o','o','o','o','v','v','v','v','v']
# for ax1_s, ax1_c, ax1_m in zip(ax1_subjects, ax1_colors, ax1_markers):
#     ax1.scatter([ir + np.random.rand(1) * 0.1 for ir, s in zip(irisRadii,subjects) if s == ax1_s], [pr for pr, s in zip(pupilRadii, subjects) if s == ax1_s], c = ax1_c, marker = ax1_m, label = ax1_s)
# ax1.set_xlabel('Iris Radius (pixels)')
# ax1.set_ylabel('Pupil Radius (pixels)')
# ax1.legend(loc = 'lower left')
# ax1.set_ylim([0,26])
# ax2 = fig.add_subplot(1,3,2)
# ax2_subjects = ['AM','CF','EL','JHK','JSK','KA','MM','MS','NK','SB','WL']
# ax2_colors = ['b','g','r','c','m','k','b','g','r','c','m']
# ax2_markers = ['o','o','o','o','o','o','v','v','v','v','v']
# for ax2_s, ax2_c, ax2_m in zip(ax2_subjects, ax2_colors, ax2_markers):
#     ax2.scatter([p + np.random.rand(1) * 0.1 for p, s in zip(pupilRadii,subjects) if s == ax2_s], [pd for pd, s in zip(np.linalg.norm(pupilDisplacements, axis = 1), subjects) if s == ax2_s], c = ax2_c, marker = ax2_m, label = ax2_s)
# ax2.set_xlabel('Pupil Radius (pixels)')
# ax2.set_ylabel('Pupil Center Displacement (pixels)')
# ax2.set_ylim([-1,6])
# ax2.legend(loc = 'lower left')
# ax3 = fig.add_subplot(1,3,3)
# ax3.set_xlabel('Intensity (pixel values 0 to 255)')
# ax3.set_ylabel('Frequency')
# ax3.set_xlim([0,255])
# bins = np.linspace(0, 300, 100)
# pupilIntensity_raw = list()
# pupilIntensity_mean = list()
# for pi in results['pupilSamples']['intensity']:
#     pupilIntensity_raw.extend(np.reshape(pi,[1,-1]))
#     pupilIntensity_mean.append(np.mean(pi))
# pupilIntensity_raw = np.squeeze(np.reshape(pupilIntensity_raw,[1,-1]))
# irisIntensity_dark_raw = list()
# irisIntensity_dark_mean = list()
# irisIntensity_bright_raw = list()
# irisIntensity_bright_mean = list()
# for ii in results['irisSamples']['intensity']:
#     irisIntensity_dark_raw.extend(np.reshape(ii[0],[1,-1]))
#     irisIntensity_dark_mean.append(np.mean(ii[0]))
#     irisIntensity_bright_raw.extend(np.reshape(ii[1],[1,-1]))
#     irisIntensity_bright_mean.append(np.mean(ii[1]))
# irisIntensity_dark_raw = np.squeeze(np.reshape(irisIntensity_dark_raw,[1,-1]))
# irisIntensity_bright_raw = np.squeeze(np.reshape(irisIntensity_bright_raw,[1,-1]))
# scleraIntensity_dark_raw = list()
# scleraIntensity_dark_mean = list()
# scleraIntensity_bright_raw = list()
# scleraIntensity_bright_mean = list()
# for si in results['scleraSamples']['intensity']:
#     scleraIntensity_dark_raw.extend(np.reshape(si[0],[1,-1]))
#     scleraIntensity_dark_mean.append(np.mean(si[0]))
#     scleraIntensity_bright_raw.extend(np.reshape(si[1],[1,-1]))
#     scleraIntensity_bright_mean.append(np.mean(si[1]))
# scleraIntensity_dark_raw = np.squeeze(np.reshape(scleraIntensity_dark_raw,[1,-1]))
# scleraIntensity_bright_raw = np.squeeze(np.reshape(scleraIntensity_bright_raw,[1,-1]))
# skinIntensity_dark_raw = list()
# skinIntensity_dark_mean = list()
# skinIntensity_bright_raw = list()
# skinIntensity_bright_mean = list()
# for si in results['skinSamples']['intensity']:
#     skinIntensity_dark_raw.extend(np.reshape(si[0],[1,-1]))
#     skinIntensity_dark_mean.append(np.mean(si[0]))
#     skinIntensity_bright_raw.extend(np.reshape(si[1],[1,-1]))
#     skinIntensity_bright_mean.append(np.mean(si[1]))
# skinIntensity_dark_raw = np.squeeze(np.reshape(skinIntensity_dark_raw,[1,-1]))
# skinIntensity_bright_raw = np.squeeze(np.reshape(skinIntensity_bright_raw,[1,-1]))
# hist_alpha = 0.5
# ax3.hist(pupilIntensity_raw, bins, alpha = hist_alpha, label = 'pupil')
# ax3.hist(irisIntensity_dark_raw, bins, alpha = hist_alpha, label = 'iris, dark')
# ax3.hist(irisIntensity_bright_raw, bins, alpha = hist_alpha, label = 'iris, bright')
# ax3.hist(scleraIntensity_dark_raw, bins, alpha = hist_alpha, label = 'sclera, dark')
# ax3.hist(scleraIntensity_bright_raw, bins, alpha = hist_alpha, label = 'sclera, bright')
# ax3.hist(skinIntensity_dark_raw, bins, alpha = hist_alpha, label = 'skin, dark')
# ax3.hist(skinIntensity_bright_raw, bins, alpha = hist_alpha, label = 'skin, bright')
# ax3.legend(loc = 'upper right')
# plt.show()

# # stat for looking left.

# # stat for looking right.

