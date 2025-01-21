import tqdm
import pandas as pd
import json


FEEDBACK_PATH = 'data/feedback.csv'
COMMENTS_PATH = 'data/student-comments.csv'

old_columns = ['task_id','task_title','assignment_id','assignment_name','course_name','course_id','user_id','student_first_name','student_last_name','data']
new_columns = [
'course_id','course_name','task_id','task_title','assignment_id','assignment_name','user_id','ta_feedback_text','category_hints','category_hint_1','category_hint_2','category_hint_3','category_hint_1_embedding','category_hint_2_embedding','category_hint_3_embedding','category_hint_idx'
]


def feedback_has_more_than_one_word(feedback):
    return len(feedback.split()) > 1

student_comments = pd.read_csv(COMMENTS_PATH)

new_feedbacks = []
for index, (i, row) in tqdm.tqdm(enumerate(student_comments.iterrows()), total=student_comments.shape[0]):
    student_responses = row['data']
    if student_responses and pd.isna(student_responses) == False:
        student_responses = json.loads(student_responses)

        if row['task_id'] == 3726:
            secondary_feedback = 'Engaging: ' + (student_responses['input1'] or '') + \
                                 ', Covered Aspects: ' + (student_responses['input2'] or '') + \
                                 ', Would Recommend: ' + (student_responses['input4'] or '') + \
                                 ', Required Rewatch: ' + (student_responses['input6'] or '')

            if student_responses['input3'] and pd.isna(student_responses['input3']) == False and feedback_has_more_than_one_word(student_responses['input3']):
                primary_feedback1 = 'Feedback upon video covering lab techniques: ' + student_responses['input3']
                new_feedbacks.append({
                    'course_id': row['course_id'],
                    'course_name': row['course_name'],
                    'task_id': row['task_id'],
                    'task_title': row['task_title'],
                    'assignment_id': row['assignment_id'],
                    'assignment_name': row['assignment_name'],
                    'student_id': row['user_id'],
                    'ta_feedback_text': primary_feedback1,
                    'secondary_feedback': secondary_feedback
                })

            if student_responses['input5'] and pd.isna(student_responses['input5']) == False and feedback_has_more_than_one_word(student_responses['input5']):
                primary_feedback2 = 'Would Recommend to Other Students Feedback: ' + student_responses['input5']
                new_feedbacks.append({
                    'course_id': row['course_id'],
                    'course_name': row['course_name'],
                    'task_id': row['task_id'],
                    'task_title': row['task_title'],
                    'assignment_id': row['assignment_id'],
                    'assignment_name': row['assignment_name'],
                    'student_id': row['user_id'],
                    'ta_feedback_text': primary_feedback2,
                    'secondary_feedback': secondary_feedback
                })

            if student_responses['input7'] and pd.isna(student_responses['input7']) == False and feedback_has_more_than_one_word(student_responses['input7']):
                primary_feedback3 = 'Potential Improvements Feedback: ' + student_responses['input7']
                new_feedbacks.append({
                    'course_id': row['course_id'],
                    'course_name': row['course_name'],
                    'task_id': row['task_id'],
                    'task_title': row['task_title'],
                    'assignment_id': row['assignment_id'],
                    'assignment_name': row['assignment_name'],
                    'student_id': row['user_id'],
                    'ta_feedback_text': primary_feedback3,
                    'secondary_feedback': secondary_feedback
                })
        else:
            primary_feedbacks = [student_responses['input4'], student_responses['input5']]
            secondary_feedback = 'Helpful: ' + (student_responses['input1'] or '') + \
                                 ', Clear Explanation: ' + (student_responses['input2'] or '') + \
                                 ', Video Speed: ' + (student_responses['input6'] or '')

            for primary_feedback in primary_feedbacks:
                if primary_feedback and pd.isna(primary_feedback) == False and feedback_has_more_than_one_word(primary_feedback):
                    new_feedbacks.append({
                        'course_id': row['course_id'],
                        'course_name': row['course_name'],
                        'task_id': row['task_id'],
                        'task_title': row['task_title'],
                        'assignment_id': row['assignment_id'],
                        'assignment_name': row['assignment_name'],
                        'student_id': row['user_id'],
                        'ta_feedback_text': primary_feedback,
                        'secondary_feedback': secondary_feedback
                    })

new_feedbacks_df = pd.DataFrame(new_feedbacks)
new_feedbacks_df.to_csv(FEEDBACK_PATH, index=False)


