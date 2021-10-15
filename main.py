from graph_cut import Graph_cut
gc = Graph_cut(mode='face',
               data_path='/data2/suyu/shunde',
               output_path='/data2/suyu/shunde',
               similarity_file_name='similarity.npy',
               edge_file_name='edge_06.npy',
               label_file_name='pred_06.npy',
               face_threshold=0.6,
               body_threshold=0.7,
               face_name='pred_06.npy',
               body_name='pred_07.npy',
               fb_name='pred_06_07.npy',
               fb_edge_path='face_body_edge.npy'
               )

gc.graph_generate()
gc.graph_cut()
# gc.face_body_combine()
gc.evaluate()
# gc.evaluate_each()