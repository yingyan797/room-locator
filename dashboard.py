from flask import Flask, render_template, request
import decision_tree as dt
import numpy as np
from visualization import Tree_Visualizer
import os
import glob

app = Flask(__name__)
decision = dt.Decision()

@app.route('/', methods=['GET', 'POST']) 
def index():
    print(request.form)
    room_nums = []
    report = []
    conf = []
    graphs = []
    attrs = ["x"+str(i) for i in range(7)]    
    file = request.form.get("dataset")
    if request.form.get("reselect"):
        decision.__init__()
        file = ""
    else:
        if file == "Clean Dataset":
            decision.load_data(dt.cleanData)
        elif file == "Noisy Dataset":
            decision.load_data(dt.noisyData)
        elif file != None:
            decision.load_data("wifi_db/"+file)
   
        if decision.all_data is not None:
            if request.form.get("cross"):
                mt, report, confmat = decision.cross_validation()
                th = ["Pred/Actl"]+[str(i+1) for i in range(decision.label_count)]
                conf.append(th)
                for i in range(decision.label_count):
                    tr = [str(i+1)]
                    for v in confmat[i]:
                        tr.append(str(int(v)))
                    conf.append(tr)
            elif request.form.get("decision"):
                decision.fit()

    if decision.decision_tree:
        if request.form.get("visualization"):
            tv = Tree_Visualizer(decision.decision_tree)
            session = get_session()
            graphs = tv.visualize(session)
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

    return render_template('index.html', data_name=file, graphs=graphs, decision=decision, 
                           attrs=attrs, room_nums=room_nums, report=report, conf=conf)

@app.route('/graphs', methods=['GET', 'POST']) 
def graphs():
    print(request.form)
    history1 = []
    history2 = []
    if request.form.get("clear"):
        files = glob.glob('static/plots/*')
        for f in files:
            os.remove(f)
        open("graphdb.csv", "w").write("")
    else:
        f = open("graphdb.csv", "r")
        left = True
        while True:
            line = f.readline()
            if line != "":
                gname = line[2:-1]
                if left:
                    history1.append(gname)
                    left = False
                else:
                    history2.append(gname)
                    left = True
            else:
                break
    return render_template('graphs.html', history1=history1, history2=history2)

def get_session():
    f = open("graphdb.csv", "r")
    lines = f.readlines()
    f.close()
    if lines == []:
        return 0
    
    num = lines[-1][0]
    if int(num) >= 8:
        f = open("graphdb.csv", "w")
        while lines[0][0] == '0':
            lines.pop(0)
            os.remove("static/plots/"+lines[0][2:-1])
        for i in range(len(lines)):
            lines[i] = str(int(lines[i][0])-1)+lines[i][1:-6]+str(int(lines[i][-6])-1)+lines[i][-5:]
        f.writelines(lines)
        f.close()
        return int(num)
    
    return int(num)+1

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)