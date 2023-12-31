from flask import Flask, render_template, request
import decision_tree as dt
import numpy as np
from visualization import Tree_Visualizer, Wifi_Visualizer
import os
import glob

app = Flask(__name__)
decision = dt.Decision()

class History:
    def __init__(self, mode):
        self.mode = mode

graphic_history = History("")


@app.route('/', methods=['GET', 'POST'])    # main page
def index():
    print(request.form)
    room_nums = []
    accuracy = None
    prf_table = None
    conf = []
    graphs = []
    wifi_visual = None
    attrs = ["x"+str(i) for i in range(7)]    
    file = request.form.get("dataset")
    if request.form.get("reselect"):
        decision.__init__()
        file = ""
    else:
        if file == "clean":
            decision.load_data(dt.cleanData)
        elif file == "noisy":
            decision.load_data(dt.noisyData)
        elif file:
            decision.load_data("wifi_db/"+file)
   
        if decision.all_data is not None:
            if request.form.get("plotdata"):
                wifivs = Wifi_Visualizer(decision)
                wifi_visual = wifivs.plot_dataset()
            elif request.form.get("clear_wdp"):
                files = glob.glob('static/wifi_visual/*')
                for f in files:
                    os.remove(f)
            prune_depth = request.form.get("prunedepth")
            prune_pct = request.form.get("prunepct")
            if prune_depth or prune_pct:
                decision.apply_pruning(prune_depth, prune_pct)
            if request.form.get("cross"):
                mt, confmat, accuracy, prf_table = decision.cross_validation()
                th = ["Actl./Pred."]+["P. Room "+str(i+1) for i in range(decision.label_count)]
                conf.append(th)
                for i in range(decision.label_count):
                    tr = ["A. Room "+str(i+1)]
                    for v in confmat[i]:
                        tr.append(str(v))
                    conf.append(tr)
            elif request.form.get("decision"):
                decision.fit()

    if decision.decision_tree:
        if request.form.get("visualization"):
            tv = Tree_Visualizer(decision.decision_tree)
            session = get_session()
            graphs = tv.visualize(session, decision.data_name)
        if request.form.get("predict"):
            data_predict = None
            predf = request.form.get("pred_file")
            if predf:
                data_predict = np.loadtxt("predictions/"+predf)
            else:
                data = []
                for attr in attrs:
                    signal = request.form.get(attr)
                    if signal is None or signal == "":
                        break
                    else:
                        data.append(float(signal))
                if len(data) == 7:
                    data_predict = np.array([data])
            if data_predict is not None:
                room_nums = dt.predict(decision.decision_tree, data_predict)

    return render_template('index.html', graphs=graphs, decision=decision, wifi_visual=wifi_visual,
                           attrs=attrs, room_nums=room_nums, prf=prf_table, accuracy=accuracy, conf=conf)

@app.route('/graphs', methods=['GET', 'POST'])  # decision tree graphic history page
def graphs():
    print(request.form)
    history1 = []
    history2 = []
    if request.form.get("clear"):
        if graphic_history.mode == "Decision tree":
            files = glob.glob('static/plots/*')
            for f in files:
                os.remove(f)
            open("graphdb.csv", "w").write("")
        else:
            files = glob.glob('static/wifi_visual/*')
            if request.form.get("clear"):
                for f in files:
                    os.remove(f)

    elif request.form.get("graphdt"):
        graphic_history.mode = "Decision tree"
        f = open("graphdb.csv", "r")
        left = True
        while True:
            line = f.readline()
            if line != "":
                segs = line.split(',')
                data_name, gname = segs[1], segs[2][:-1]
                if left:
                    history1.append((data_name,gname))
                    left = False
                else:
                    history2.append((data_name,gname))
                    left = True
            else:
                break
    elif request.form.get("graphdata"):
        graphic_history.mode = "Dataset"
        left = True
        files = glob.glob('static/wifi_visual/*')
        for f in files:
            if left:
                history1.append(f)
                left = False
            else:
                history2.append(f)
                left = True
            
    return render_template('graphs.html', mode=graphic_history.mode, history1=history1, history2=history2)

def get_session():      # For naming decision tree graphs
    f = open("graphdb.csv", "r")
    lines = f.readlines()
    f.close()
    if lines == []:
        return 0
    
    num = lines[-1][0]
    if int(num) >= 8:
        f = open("graphdb.csv", "w")
        while lines[0][0] == '0':
            gname = lines.pop(0).split(',')[2][:-1]
            os.remove("static/plots/"+gname)
        for i in range(len(lines)):
            lines[i] = str(int(lines[i][0])-1)+lines[i][1:-6]+str(int(lines[i][-6])-1)+lines[i][-5:]
        f.writelines(lines)
        f.close()
        return int(num)
    
    return int(num)+1


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)