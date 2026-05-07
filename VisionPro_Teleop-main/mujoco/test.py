import mujoco

def validate_xml(xml_path):
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        print("✅ XML 文件合法，模型加载成功！")
        return model
    except Exception as e:
        print("❌ XML 文件存在错误:")
        print(e)
        return None

# 示例调用
model = validate_xml("assets/scene/scene_leap_right.xml")