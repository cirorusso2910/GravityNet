import os
import shutil

from natsort import natsort

from net.utility.msg.msg_output_complete import msg_output_complete


def resume_output_validation(experiment_ID: str,
                             output_resume_path: str,
                             output_path: str):
    """
    Resume output validation

    :param experiment_ID: experiment ID
    :param output_resume_path: output path from resume experiment
    :param output_path: output path from experiment
    """

    output_list_file = sorted(os.listdir(output_resume_path))
    output_list_file = natsort.natsorted(output_list_file)

    tot_file = len(output_list_file)

    for i in range(tot_file):
        prefix_output = output_list_file[i].split('|')[0]

        old_output_path = os.path.join(output_resume_path, output_list_file[i])
        new_output_filename = prefix_output + "|" + experiment_ID + ".png"
        new_output_path = os.path.join(output_path, new_output_filename)

        # copy
        shutil.copy(src=old_output_path,
                    dst=new_output_path)

    msg_output_complete(output_type='resume validation')
