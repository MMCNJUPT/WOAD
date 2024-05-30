import numpy as np
import d3rlpy
import onnxruntime as rt
import pandas as pd

excel_file = "./RL_data/data.xlsx"  # 将此处替换为您的Excel文件路径
df = pd.read_excel(excel_file)

all_columns_data = df.values.T
features = all_columns_data[:, 0:6]
features = features / [420, 420, 420, 420, 400, 400]
sparse_reward = all_columns_data[:, 7]

classifier = rt.InferenceSession('./RL_data/classifier_model.onnx')
output_name = []
input_name = classifier.get_inputs()[0].name
print(input_name)
for i in range(len(classifier.get_outputs())):
    # print(i, sess_fl32.get_outputs()[i].name)
    name = str(classifier.get_outputs()[i].name)
    output_name.append(name)
print(output_name)

sess_fl32 = rt.InferenceSession('./RL_data/DSAC.onnx')
output_name_fl32 = []
input_name_fl32 = sess_fl32.get_inputs()[0].name
print(input_name_fl32)
for i in range(len(sess_fl32.get_outputs())):
    # print(i, sess_fl32.get_outputs()[i].name)
    name = str(sess_fl32.get_outputs()[i].name)
    output_name_fl32.append(name)

observations = np.load('./RL_data/state.npy')
observations = observations[:, :]
observations = np.swapaxes(observations, 0, 1)
terminals = np.load('./RL_data/terminals.npy')
print(observations.shape, terminals.shape)

actions = np.load('./RL_data/new_action.npy')
reward = np.zeros(shape=(328, 1))
observations = observations[:, :] / [420, 420, 420, 420, 400, 400]
for i in range(observations.shape[0]):
    # observations_da = observations[i, 4:].reshape((1, 2)).astype(np.float32)
    # print(observations_da)
    # predict_value = sess_fl32.run(output_name_fl32, {input_name_fl32: observations_da})
    # actions_now = predict_value[len(predict_value)-2]
    # print(actions_now)
    # actions[i, :] = actions_now
    reward_xiao = 0
    observations_xiao = features[i, :].reshape((1, 6)).astype(np.float32)
    predict_value_xiao = classifier.run(output_name, {input_name: observations_xiao})
    # print(predict_value_xiao)
    # exit()
    dui = predict_value_xiao[0][0][int(actions[i])]
    dong = predict_value_xiao[1][0][int(actions[i])]
    budui = predict_value_xiao[0][0][int(1 - actions[i])]
    jing = predict_value_xiao[1][0][int(1 - actions[i])]
    # print(i, predict_value_xiao[0][0][int(1-actions[i])], predict_value_xiao[1][0][int(1-actions[i])])
    print(observations_xiao)
    if 0 < observations_xiao[0][4] <= 0.75:
        if actions[i] == 0:
            reward_xiao = dui + dong - budui - jing
        if actions[i] == 1:
            reward_xiao = budui + jing - dui - dong
    else:
        if actions[i] == 0:
            reward_xiao = -1
        if actions[i] == 1:
            reward_xiao = budui + jing - dui - dong
    print(actions[i], reward_xiao)
    # reward_xiao = predict_value_xiao[0][0][int(1 - actions[i])] - 0.5
    # print(reward_xiao, actions[i])
    reward[i, :] = sparse_reward[i] + reward_xiao

# print(actions)
reward = np.squeeze(reward, axis=1)
# reward = np.zeros(shape=actions.shape) + xiaonao(actions_next, observations_now)

print(observations.shape)
print(terminals.shape)
print(reward.shape)
print(terminals.shape)

dataset = d3rlpy.dataset.MDPDataset(
    observations=observations,
    actions=actions,
    rewards=reward,
    terminals=terminals,
)
dataset.dump('./RL_data/cerebellar_dataset_RL.h5')
