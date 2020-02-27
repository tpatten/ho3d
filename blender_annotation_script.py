import os
import bpy
import numpy as np
import pickle
import transforms3d as tf3d
import cv2

c_id = np.array([0])
target_dir = "/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/train/ABF10/meta/"
gripper_model_dir = "/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/"
gripper_model_fn = "hand_open_aligned.ply"
textured_model_dir = "/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/models/"
cut_z = 1.5
textured_obj_name = None
hand_mat = bpy.data.materials.new(name="hand_material")
hand_mat.diffuse_color = (0.5, 0.376, 0.283)


class loadNext(bpy.types.Operator):
    bl_idname = "object.load_next"
    bl_label = "Load Next file"

    def execute(self, context):
        global textured_obj_name, hand_mat

        ind = c_id[0]

        # Save the last pose
        if ind > 0:
            obj = bpy.data.objects[obj_name]
            pose = np.zeros((4, 4))
            pose[:, :] = obj.matrix_world
            pose = pose.reshape(-1)
            grasp_pose = pose

            file_name = file_list[c_id[0] - 1]
            file_name = file_name.replace("cloud", "grasp_bl")
            file_name = file_name.replace("ply", "pkl")
            gt_fn = os.path.join(target_dir, file_name)
            # print(grasp_pose)

            with open(gt_fn, 'wb') as f:
                pickle.dump(grasp_pose, f)
            print("saved file to ", gt_fn)

        # Now load new data
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                if obj.name == obj_name or (textured_obj_name is not None and obj.name == textured_obj_name):
                    obj.select = False
                    obj.hide = False
                else:
                    obj.hide = False
                    obj.select = True
        bpy.ops.object.delete()
        if ind >= len(file_list):
            return {'FINISHED'}

        target_fn = os.path.join(target_dir, file_list[ind])
        _ = bpy.ops.import_mesh.ply(filepath=target_fn)

        for o in bpy.context.scene.objects:
            if o.type == 'MESH':
                if o.name == obj_name:
                    o.select = True
                else:
                    o.select = False

        for a in bpy.context.screen.areas:
            if a.type == 'VIEW_3D':
                for s in a.spaces:
                    if s.type == 'VIEW_3D':
                        s.show_manipulator = True
                        s.transform_manipulators = {'TRANSLATE', 'ROTATE'}

        # Load and transform the gripper
        if ind == 0:
            grasp_file_name = file_list[ind]
            grasp_file_name = grasp_file_name.replace("cloud", "grasp")
            grasp_file_name = grasp_file_name.replace("ply", "pkl")
            grasp_file_name = os.path.join(target_dir, grasp_file_name)
            in_pose = None
            with open(grasp_file_name, 'rb') as f:
                try:
                    in_pose = pickle.load(f, encoding='latin1')
                except:
                    in_pose = pickle.load(f)
            if in_pose is not None:
                euler_angles = tf3d.euler.mat2euler(in_pose[:3, :3])
                obj_object = bpy.data.objects[obj_name]
                obj_object.rotation_euler.x = euler_angles[0]
                obj_object.rotation_euler.y = euler_angles[1]
                obj_object.rotation_euler.z = euler_angles[2]
                obj_object.location.x = in_pose[0, 3]
                obj_object.location.y = in_pose[1, 3]
                obj_object.location.z = in_pose[2, 3]
            else:
                print("Did not find grasp pose to load")
        else:
            # Add tiny bit of noise
            angle_range = [-0.05, 0.05]
            trans_range = [-0.01, 0.01]
            obj_object = bpy.data.objects[obj_name]
            obj_object.rotation_euler.x += np.random.uniform(angle_range[0], angle_range[1])
            obj_object.rotation_euler.y += np.random.uniform(angle_range[0], angle_range[1])
            obj_object.rotation_euler.z += np.random.uniform(angle_range[0], angle_range[1])
            obj_object.location.x += np.random.uniform(trans_range[0], trans_range[1])
            obj_object.location.y += np.random.uniform(trans_range[0], trans_range[1])
            obj_object.location.z += np.random.uniform(trans_range[0], trans_range[1])

        # Load and transform the object model
        anno_file_name = file_list[ind]
        anno_file_name = anno_file_name.replace("cloud_", "")
        anno_file_name = anno_file_name.replace("ply", "pkl")
        anno_file_name = os.path.join(target_dir, anno_file_name)
        object_textured_name = None
        object_textured_rot = None
        object_textured_tra = None
        with open(anno_file_name, 'rb') as f:
            try:
                pickle_data = pickle.load(f, encoding='latin1')
            except:
                pickle_data = pickle.load(f)
            object_textured_name = pickle_data['objName']
            object_textured_rot = pickle_data['objRot']
            object_textured_tra = pickle_data['objTrans']
        if object_textured_name is not None:
            if textured_obj_name is None:
                target_fn = os.path.join(textured_model_dir, object_textured_name, "textured_simple.obj")
                _ = bpy.ops.import_scene.obj(filepath=target_fn)
                textured_object = bpy.context.selected_objects[0]
                textured_obj_name = textured_object.name
                print('Imported name: ', textured_obj_name)
            textured_object = bpy.data.objects[textured_obj_name]
            textured_object.location.x = object_textured_tra[0]
            textured_object.location.y = object_textured_tra[1]
            textured_object.location.z = object_textured_tra[2]
            euler_angles = tf3d.euler.mat2euler(cv2.Rodrigues(object_textured_rot)[0])
            textured_object.rotation_euler.x = euler_angles[0]
            textured_object.rotation_euler.y = euler_angles[1]
            textured_object.rotation_euler.z = euler_angles[2]
        else:
            print("Did not find textured model to load")

        # Load the hand mesh
        hand_file_name = file_list[ind]
        hand_file_name = hand_file_name.replace("cloud", "hand_mesh")
        hand_file_name = os.path.join(target_dir, hand_file_name)
        _ = bpy.ops.import_mesh.ply(filepath=hand_file_name)
        hand_name = bpy.context.selected_objects[0].name
        hand_mesh = bpy.data.objects[hand_name]
        hand_mesh.data.materials.append(hand_mat)

        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                obj.hide = False
                if obj.name == obj_name:
                    obj.select = True
                else:
                    obj.select = False

        c_id[0] = c_id[0] + 1
        return {'FINISHED'}


class resetObj(bpy.types.Operator):
    bl_idname = "object.reset"
    bl_label = "Reset Pose"

    def execute(self, context):
        obj_object = bpy.data.objects[obj_name]
        obj_object.rotation_euler.x = 0
        obj_object.rotation_euler.y = 0
        obj_object.rotation_euler.z = 0
        obj_object.location.x = 0
        obj_object.location.y = 0
        obj_object.location.z = 0
        return {'FINISHED'}

'''
class SaveFile(bpy.types.Operator):
    bl_idname = "object.savefile"
    bl_label = "Save File"

    def execute(self, context):
        scene = bpy.context.scene
        obj = bpy.data.objects[obj_name]
        pose = np.zeros((4, 4))
        pose[:, :] = obj.matrix_world
        pose = pose.reshape(-1)
        grasp_pose = pose

        file_name = file_list[c_id[0] - 1]
        file_name = file_name.replace("cloud", "grasp_bl")
        file_name = file_name.replace("ply", "pkl")
        gt_fn = os.path.join(target_dir, file_name)
        print(grasp_pose)

        with open(gt_fn, 'wb') as f:
            pickle.dump(grasp_pose, f)
        print("saved file to ", gt_fn)

        return {'FINISHED'}
'''


class AnnotationPanel(bpy.types.Panel):
    bl_label = "Pose Annotation"
    bl_idname = "3D_VIEW_TS_bricks"
    bl_space_type = "VIEW_3D"
    bl_category = "Annotation"
    bl_region_type = "TOOLS"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.alignment = 'CENTER'
        row.operator("object.reset")

        #row = layout.row()
        #row.alignment = 'CENTER'
        #row.operator("object.savefile")

        row = layout.row()
        row.alignment = 'CENTER'
        row.operator("object.load_next")


def register():
    bpy.utils.register_class(loadNext)
    bpy.utils.register_class(resetObj)
    #bpy.utils.register_class(SaveFile)
    bpy.utils.register_class(AnnotationPanel)


def unregister():
    bpy.utils.unregister_class(loadNext)
    bpy.utils.unregister_class(resetObj)
    #bpy.utils.unregister_class(SaveFile)
    bpy.utils.unregister_class(AnnotationPanel)


if __name__ == "__main__":
    # Delete all object first
    for o in bpy.context.scene.objects:
        if o.type == 'MESH':
            o.select = True
    bpy.ops.object.delete()

    gripper_model = os.path.join(gripper_model_dir, gripper_model_fn)
    _ = bpy.ops.import_mesh.ply(filepath=gripper_model)
    obj_name = gripper_model_fn.replace(".ply", "")
    gripper_mesh = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            if obj.name == obj_name:
                gripper_mesh = obj
    mat = bpy.data.materials.new(name="gripper_material")
    gripper_mesh.data.materials.append(mat)
    mat.diffuse_color = (0.016, 0.311, 0.016)

    index = 0
    file_list = []
    grasp_pose = None
    for file in sorted(os.listdir(target_dir)):
        if file.startswith("cloud"):
            file_list.append(file)

    register()
