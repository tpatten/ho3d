import os
import bpy, bmesh
import numpy as np
import pickle
import transforms3d as tf3d
import cv2

target_dir = "/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/train/"
gripper_model_dir = "/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/"
textured_model_dir = "/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/models/"
cut_z = 1.5
sequence_name1 = None
sequence_name2 = None

sequence1 = "ABF10"
sequence2 = "ABF11"

hand_mat1 = bpy.data.materials.new(name="hand_material1")
hand_mat1.diffuse_color = (0.5, 0.376, 0.283)
hand_mat2 = bpy.data.materials.new(name="hand_material2")
hand_mat2.diffuse_color = (0.016, 0.311, 0.016)


class loadFiles(bpy.types.Operator):
    bl_idname = "object.load_files"
    bl_label = "Load files"

    def execute(self, context):
        global sequence_name1, sequence_name2, hand_mat1, hand_mat2

        for a in bpy.context.screen.areas:
            if a.type == 'VIEW_3D':
                for s in a.spaces:
                    if s.type == 'VIEW_3D':
                        s.show_manipulator = True
                        s.transform_manipulators = {'TRANSLATE', 'ROTATE'}

        # # Sequence 1
        file_name = "cloud_0000.ply"
        sequence_name = sequence1
        hand_mat = hand_mat1

        # Load and transform the object model
        anno_file_name = file_name
        anno_file_name = anno_file_name.replace("cloud_", "")
        anno_file_name = anno_file_name.replace("ply", "pkl")
        anno_file_name = os.path.join(target_dir, sequence_name, "meta", anno_file_name)
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
            target_fn = os.path.join(textured_model_dir, object_textured_name, "textured_simple.obj")
            _ = bpy.ops.import_scene.obj(filepath=target_fn)
            textured_object = bpy.context.selected_objects[0]
            print('Imported name: ', textured_object.name)
            textured_object = bpy.data.objects[textured_object.name]
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
        hand_file_name = file_name
        hand_file_name = hand_file_name.replace("cloud", "hand_mesh")
        hand_file_name = os.path.join(target_dir, sequence_name, "meta", hand_file_name)
        _ = bpy.ops.import_mesh.ply(filepath=hand_file_name)
        hand_name = bpy.context.selected_objects[0].name
        hand_mesh = bpy.data.objects[hand_name]
        hand_mesh.data.materials.append(hand_mat)

        # Join the meshes
        ctx = bpy.context.copy()
        ctx['active_object'] = hand_mesh
        ctx['selected_objects'] = textured_object
        bpy.ops.object.join(ctx)
        selected_object = bpy.context.selected_objects[0]
        sequence_name1 = selected_object.name

        # # Sequence 2
        sequence_name = sequence2
        hand_mat = hand_mat2

        # Load and transform the object model
        anno_file_name = file_name
        anno_file_name = anno_file_name.replace("cloud_", "")
        anno_file_name = anno_file_name.replace("ply", "pkl")
        anno_file_name = os.path.join(target_dir, sequence_name, "meta", anno_file_name)
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
            target_fn = os.path.join(textured_model_dir, object_textured_name, "textured_simple.obj")
            _ = bpy.ops.import_scene.obj(filepath=target_fn)
            textured_object = bpy.context.selected_objects[0]
            print('Imported name: ', textured_object.name)
            textured_object = bpy.data.objects[textured_object.name]
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
        hand_file_name = file_name
        hand_file_name = hand_file_name.replace("cloud", "hand_mesh")
        hand_file_name = os.path.join(target_dir, sequence_name, "meta", hand_file_name)
        _ = bpy.ops.import_mesh.ply(filepath=hand_file_name)
        hand_name = bpy.context.selected_objects[0].name
        hand_mesh = bpy.data.objects[hand_name]
        hand_mesh.data.materials.append(hand_mat)

        # Join the meshes
        ctx = bpy.context.copy()
        ctx['active_object'] = hand_mesh
        ctx['selected_objects'] = textured_object
        bpy.ops.object.join(ctx)
        selected_object = bpy.context.selected_objects[0]
        sequence_name2 = selected_object.name

        return {'FINISHED'}


class printPose(bpy.types.Operator):
    bl_idname="object.print_pose"
    bl_label = "Print pose"

    def execute(self, context):
        global sequence_name2
        obj = bpy.data.objects[sequence_name2]
        pose = np.zeros((4, 4))
        pose[:, :] = obj.matrix_world
        print(pose)

        return {'FINISHED'}


class saveTransform(bpy.types.Operator):
    bl_idname="object.save_transform"
    bl_label = "Save transform"

    def execute(self, context):
        global sequence_name2
        obj = bpy.data.objects[sequence_name2]
        pose = np.zeros((4, 4))
        pose[:, :] = obj.matrix_world
        print(pose)

        save_file_name = os.path.join(target_dir, sequence2, "meta", "transformation_init.pkl")
        with open(save_file_name, 'wb') as f:
            pickle.dump(pose, f)

        return {'FINISHED'}


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
        row.operator("object.load_files")
        row = layout.row()
        row.alignment = 'CENTER'
        row.operator("object.print_pose")
        row = layout.row()
        row.alignment = 'CENTER'
        row.operator("object.save_transform")


def register():
    bpy.utils.register_class(loadFiles)
    bpy.utils.register_class(printPose)
    bpy.utils.register_class(saveTransform)
    bpy.utils.register_class(AnnotationPanel)


def unregister():
    bpy.utils.unregister_class(loadFiles)
    bpy.utils.unregister_class(printPose)
    bpy.utils.unregister_class(saveTransform)
    bpy.utils.unregister_class(AnnotationPanel)


if __name__ == "__main__":
    # Delete all object first
    for o in bpy.context.scene.objects:
        if o.type == 'MESH':
            o.select = True
    bpy.ops.object.delete()

    register()
