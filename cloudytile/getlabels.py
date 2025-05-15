## ======= ##
## IMPORTS ##
## ======= ##
import labelbox
import json
import pandas as pd

## =================================== ##
## IMPORT LABELS FROM LABELBOX VIA API ##
## =================================== ##

LB_API_KEY = ''
client = labelbox.Client(api_key = LB_API_KEY)
export_task = labelbox.ExportTask.get_task(client, "cmapn9m37041i07xlcfjp0yvf")
export_json = [dr.json for dr in export_task.get_buffered_stream()]



## ======================================= ##
## CREATE DF WIH LABELS AND .png FILENAMES ##
## ======================================= ##

project_uid = "cmapjzjp200td0704hly95bb8"  # your real project UID

rows = []
for rec in export_json:
    # 1) grab the PNG filename
    filename = rec["data_row"]["external_id"]
    
    # 2) loop over each label (usually a single entry in your case)
    for lbl in rec["projects"][project_uid]["labels"]:
        # 3) loop over each classification answer
        for cls in lbl["annotations"]["classifications"]:
            # pull out the yes/no
            answer = cls["radio_answer"]["name"]
            rows.append({
                "filename": filename,
                "lb_label":    answer
            })

final_df = pd.DataFrame(rows)
final_df['label_num'] = final_df['lb_label'].map({'no': 0, 'yes': 1})
print(final_df.head())
