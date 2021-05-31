import json

def annotation():

    # 与えられているアノテーションデータは矩形ではないため、繋がっている座標のものを
    # たとえ建物の途中であっても一つの新築と見做して矩形にアノテーションする
    with open('../../data/buildings/readme/submit.json') as f:
        anno_data = json.load(f)

    images_dict = {}
    for image_num, pos_data in anno_data.items():
        result_data = []
        for pos in pos_data.values():
            tmp_data = []
            for y_pos,x_poses in pos.items():
                # 途中でpopするとindexがズレるのでpopする番号を保存しておく
                pop_nums = []
                # すでに発見された建物の終了を判定
                for index, tmp_pos in enumerate(tmp_data):
                    if tmp_pos[0] not in x_poses:
                        # y座標の終わりを追記
                        tmp_pos[1].append(int(y_pos))
                        # pop_numに追加
                        pop_nums.append(index)

                # 逆からやらないとpop番号がズレるのでreverseする
                for pop_num in reversed(pop_nums):
                    result_data.append(tmp_data.pop(pop_num))
                
                # 新規の新築の建物の追加フェイズ
                for x_pos in x_poses:
                    # tmp_dataではy座標も保持しているのでx座標のみを1時的に取り出す
                    tmp_x_pos = [x[0] for x in tmp_data]
                    if x_pos not in tmp_x_pos:
                        tmp_data.append([x_pos,[int(y_pos)]])

        images_dict.update({image_num:result_data})

    with open('../../data/buildings/readme/annotation.json', 'w') as f:
        json.dump(images_dict,f,indent=4)
                    
if __name__=='__main__':
    annotation()