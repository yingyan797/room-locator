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
    room_num = None
    report = []
    graphs = []
    attrs = ["x"+str(i) for i in range(7)]    
    file = request.form.get("dataset")
    if request.form.get("reselect"):
        decision.__init__()
        file = ""
    else:
        match file:
            case "clean":
                decision.load_data(dt.cleanData)
            case "noisy":
                decision.load_data(dt.noisyData)
            case "customized":
                pass
        if decision.all_data is not None:
            if request.form.get("cross"):
                report = decision.cross_validation()[1]
            else:
                decision.fit()

    if decision.decision_tree:
        if request.form.get("visualization"):
            tv = Tree_Visualizer(decision.decision_tree)
            session = get_session()
            graphs = tv.visualize(session)
        if request.form.get("predict"):
            data = []
            for attr in attrs:
                signal = request.form.get(attr)
                if signal is None or signal == "":
                    break
                else:
                    data.append(float(signal))
            if len(data) == 7:
                data_predict = np.array([data])
                room_num = int(dt.predict(decision.decision_tree, data_predict)[0])

    return render_template('index.html', data_name=file, graphs=graphs, decision=decision, 
                           attrs=attrs, room_num=room_num, report=report)

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