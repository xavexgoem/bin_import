bl_info = {
    'name': 'Dark Bin Loader',
    'version': (1, 1),
    'blender': (4, 0, 2),
    'location': 'File > Import-Export',
    'description': 'Import bin files and textures',
    'category': 'Import-Export'
}

import bpy
from mathutils import Vector, Matrix
import os

from struct import pack, unpack, calcsize
from math import pi as PI

MODEL_HEADER_SIZE = 132
class Model: 
    def __init__(self):
        self.subobjects = []
        self.materials = {} # {slot: mat}
        self.uvs = []
        self.points = []
        self.lights = []
        self.normals = []
        self.polys = []

        # as in the mds.h spec          type        

        self.signature = ""             # char[4]   
        self.version = 0                # u32       
        self.name = ""                  # char[8]  

        self.bounding_radius = 0.0      # f32       
        self.max_poly_radius = 0.0      # f32       
        self.bounds_max = 0             # f32[3]    
        self.bounds_min = 0             # f32[3]    
        self.center = 0                 # f32[3]  

        self.num_polys = 0              # u16       
        self.num_points = 0             # u16       
        self.num_parms = 0              # u16       
        self.num_materials = 0          # u8        
        self.num_calls = 0              # u8        
        self.num_vhots = 0              # u8        
        self.num_subobjects = 0         # u8   

        self.subobjects_offset = 0      # u32       
        self.materials_offset = 0       # u32       
        self.uvs_offset = 0             # u32       
        self.vhots_offset = 0           # u32       
        self.points_offset = 0          # u32       
        self.lights_offset = 0          # u32       
        self.normals_offset = 0         # u32       
        self.polys_offset = 0           # u32       
        self.nodes_offset = 0           # u32     

        self.size_bytes = 0             # u32

        self.material_ex_flags = 0      # u32       
        self.material_ex_offset = 0     # u32       
        self.material_ex_size = 0       # u32       
        
        # we never use these, but here they are anyway:
        self.mesh_off = 0               # u32
        self.submeshlist_off = 0        # u32
        self.meshes = 0                 # u16

SUB_NONE = 0x0
SUB_ROT = 0x1
SUB_SLIDE = 0x2
class Subobject: 
    def __init__(self):
        self.polys = []
        self.vhots = []
        self.parent = None
        self.child = None
        self.sibling = None

        self.points = []
        self.uvs = []
        self.lights = []
        self.normals = []

        self.vhotsblob = b""
        self.pointsblob = b""
        self.uvsblob = b""
        self.lightsblob = b""
        self.normalsblob = b""
        self.nodesblob = b""

        self.polyblobs = []
        self.polyoffsets = []

# Material types
MAT_TMAP = 0x0
MAT_COLOR = 0x1
class Material: 
    def __init__(self):
        self.name = ""
        self.type = 0x0
        self.slot = 0
        self.argb = 0x00000000 
        self.double_sided = False
        self.transparency = 0.0
        self.self_illum = 0.0

class Vhot: 
    def __init__(self):
        self.id = 0
        self.offset = Vector()

class Light: 
    def __init__(self):
        self.normal = Vector()

class Poly: 
    def __init__(self):
        self.points = []
        self.lights = []
        self.uvs = []

################ IMPORT STUFF #####################

# A wrapper I already regret:
class Funpack:
    def __init__(self, f):
        self.f = f
    def seek(self, offset):
        self.f.seek(offset)
    def tell(self):
        return self.f.tell()
    def unpack(self, fmt):
        result = unpack(fmt, self.f.read(calcsize(fmt)))
        if len(result) == 1:
            return result[0]
        else:
            return result
    def str(self, length):
        result = self.unpack("<" + str(length) + "s")
        result = result.decode('ascii')
        result = result.rstrip('\00')
        return result
    def i32(self, num = 1):
        return self.unpack("<" + str(num) + "i")
    def u32(self, num = 1):
        return self.unpack("<" + str(num) + "I")
    def i16(self, num = 1):
        return self.unpack("<" + str(num) + "h")
    def u16(self, num = 1):
        return self.unpack("<" + str(num) + "H")
    def i8(self, num = 1):
        return self.unpack("<" + str(num) + "b")
    def u8(self, num = 1):
        return self.unpack("<" + str(num) + "B")
    def r32(self, num = 1):
        return self.unpack("<" + str(num) + "f")

# Try to keep this function agnostic to Blender's API
def bin2intermediate(f):

    f = Funpack(f)

    signature = f.str(4)
    if signature != "LGMD":
        raise ValueError("Incorrect signature for static .bin file")

    model = Model()

    ##### Model header #####
    model.version = f.u32()
    model.name = f.str(8)
    model.bounding_radius = f.r32()
    model.max_poly_radius = f.r32()
    model.bounds_max = Vector(f.r32(3))
    model.bounds_min = Vector(f.r32(3))
    model.center = Vector(f.r32(3))

    model.num_polys = f.u16()
    model.num_points = f.u16()
    model.num_parms = f.u16()
    model.num_materials = f.u8()
    model.num_calls = f.u8()
    model.num_vhots = f.u8()
    model.num_subobjects = f.u8()

    model.subobjects_offset = f.u32()
    model.materials_offset = f.u32()
    model.uvs_offset = f.u32()
    model.vhots_offset = f.u32()
    model.points_offset = f.u32()
    model.lights_offset = f.u32()
    model.normals_offset = f.u32()
    model.polys_offset = f.u32()
    model.nodes_offset = f.u32()

    model.size_bytes = f.u32()

    if model.version == 4:
        model.material_ex_flags = f.u32()
        model.material_ex_offset = f.u32() 
        model.material_ex_size = f.u32()


    ##### Subobjects #####
    f.seek(model.subobjects_offset)
    for _ in range(model.num_subobjects):
        subobject = Subobject()

        subobject.name = f.str(8)
        subobject.type = f.u8()
        subobject.parm = f.i32()

        subobject.min_range = f.r32()
        subobject.max_range = f.r32()
        r1 = f.r32(3) + (0,)
        r2 = f.r32(3) + (0,)
        r3 = f.r32(3) + (0,)
        r4 = f.r32(3) + (0,)
        # r1 = r1 + (0,)
        # r2 = r2 + (0,)
        # r3 = r3 + (0,)
        # r4 = r4 + (0,)
        subobject.transform = Matrix([r1,r2,r3,r4])

        subobject.child = f.i16()
        subobject.sibling = f.i16()

        subobject.vhot_start, subobject.vhot_num = f.u16(2)
        subobject.point_start, subobject.point_num = f.u16(2)
        subobject.light_start, subobject.light_num = f.u16(2)
        subobject.norm_start, subobject.norm_num = f.u16(2)
        subobject.node_start, subobject.node_num = f.u16(2)
   
        model.subobjects.append(subobject)

    ## subobject relationships
    for i,subobject in enumerate(model.subobjects):
        if subobject.child != -1:
            model.subobjects[subobject.child].parent = subobject
        if subobject.sibling != -1:
            # set our sibling's parent to be our parent
            # NOTE it's conceivable that we haven't been assigned a parent yet
            # I haven't seen parents come after children, though
            model.subobjects[subobject.sibling].parent = subobject.parent 

    ##### Materials #####
    f.seek(model.materials_offset)
    for _ in range(model.num_materials):
        material = Material()
        material.name = f.str(16)
        material.type = f.u8()
        material.slot = f.i8()

        # If this is MAT_COLOR, we get the rgb here
        # (We need to read past these anyway if MAT_TEXTURE)
        # (And we don't care if they're junk)
        material.blue = f.u8()
        material.green = f.u8()
        material.red = f.u8()
        f.u8()  # pad
        f.u32() # pad

        print(material.name)
        model.materials[material.slot] = material

    if model.version == 4:
        f.seek(model.material_ex_offset)
        for material in model.materials.values():
            material.transparency = f.r32()
            material.self_illum = f.r32()
            if model.material_ex_size > 8:
                f.r32(2) # max texel u,v (we don't care)

    ##### UVs #####
    f.seek(model.uvs_offset)
    while f.tell() + 8 <= model.vhots_offset:
        uvx = f.r32()
        uvy = f.r32()
        model.uvs.append((uvx, uvy*-1+1)) # flipped UV y for blender

    ##### Vhots #####
    f.seek(model.vhots_offset)
    for i in range(model.num_vhots):
        vhot = Vhot()
        vhot.id = f.u32()
        vhot.offset = Vector(f.r32(3))
        for subobject in model.subobjects:
            if i >= subobject.vhot_start and i < subobject.vhot_start + subobject.vhot_num:
                subobject.vhots.append(vhot)
                break

    ##### Points #####
    f.seek(model.points_offset)
    for _ in range(model.num_points):
        model.points.append(f.r32(3))

    ##### Lights (actually vertex normals but whatever) #####
    f.seek(model.lights_offset)
    while f.tell() + 8 <= model.normals_offset:
        light = Light()

        light.mat_idx = f.u16()
        light.point_idx = f.u16()

        packed_normal = f.u32()
        light.normal.x = ((packed_normal>>16) & 0xFFC0) / 16384.0
        light.normal.y = ((packed_normal>>6)  & 0xFFC0) / 16384.0
        light.normal.z = ((packed_normal<<4)  & 0xFFC0) / 16384.0

        model.lights.append(light)

    ##### Normals #####
    f.seek(model.normals_offset)
    while f.tell() + 12 <= model.polys_offset:
        model.normals.append(f.r32(3))

    ##### Polys #####
    f.seek(model.polys_offset)
    for _ in range(model.num_polys):
        poly = Poly()

        poly.id = f.u16()
        poly.mat_id = f.u16()
        poly.type = f.u8()
        poly.n_points = f.u8()
        poly.norm = f.u16()
        poly.plane = f.r32()

        poly_subobj = None
        for __ in range(poly.n_points):
            point_idx = f.u16()

            # Go through every face's points and determine if those
            # points are within the range of points for the given subobject.
            # if _any_ of the face's points are in the range,
            # consider the whole face to be part of the subobject
            for subobject in model.subobjects:
                if (point_idx >= subobject.point_start) and (point_idx < (subobject.point_start + subobject.point_num)):
                    poly_subobj = subobject
                    break

            # in blender, polys index points in their own subobject, not the global list of points,
            # so the indices must be adjusted down
            poly.points.append(point_idx - poly_subobj.point_start)

        # unwind the other way
        #poly.points.reverse()

        poly_subobj.polys.append(poly)

        for __ in range(poly.n_points):
            poly.lights.append(f.u16())

        # TODO RGB material
        if poly.type & 3 == 3: # if this is a texture mapped poly
            for __ in range(poly.n_points):
                idx = f.u16()
                poly.uvs.append(model.uvs[idx])
            #poly.uvs.reverse() # since we reversed the points

        if model.version == 4:
            poly.mat_idx = f.u8()

        model.polys.append(poly)

    # hilariously inefficient double-sided check
    for subobject in model.subobjects:
        for i, p1 in enumerate(subobject.polys):
            sort1 = sorted(p1.points)
            for j, p2 in enumerate(subobject.polys):
                if i == j:
                    continue
                else:
                    if sort1 == sorted(p2.points):
                        model.materials[p1.mat_id].double_sided = True

    return model


#### TODO 
# 1. Try loading through Blender's image lib, in case of png
# 2. Else, try loading our own gif
# 3. ???
# 4. etc
def import_bin(context, filepath, do_search, use_empties):
    f = open(filepath, 'rb')
    m = bin2intermediate(f)

    ##### Blenderize Materials #####


    pwd = os.path.dirname(filepath)
    for mat in m.materials.values():
        if mat.name not in bpy.data.materials:

            bmat = bpy.data.materials.new(mat.name)
            bmat.use_nodes = True
            bmat.use_backface_culling = not mat.double_sided

            nodes = bmat.node_tree.nodes
            nodes.remove(nodes.get('Principled BSDF'))
            shader_node = nodes.new(type='ShaderNodeBsdfDiffuse')
            output_node = nodes.get('Material Output')

            links = nodes.data.links
            links.new(shader_node.outputs[0], output_node.inputs[0])

            if mat.type == MAT_TMAP:

                # Set up a texture node
                nodes.new("ShaderNodeTexImage")
                tex_node = nodes.get("Image Texture")
                links = bmat.node_tree.links
                links.new(tex_node.outputs[0], shader_node.inputs[0])

                # Try to find it in directories
                mat.name = mat.name.rstrip("\00")
                imgpath = bpy.path.resolve_ncase(pwd + "/" + mat.name)
                imgfile = None
                try:
                    imgfile = open(imgpath, "rb")
                except FileNotFoundError:
                    print("couldn't load from " + imgpath)
                    if(do_search):
                        directories = ["/txt/", "/txt16/"]
                        for d in directories:
                            imgpath = bpy.path.resolve_ncase(pwd + d + mat.name)
                            try:
                                imgfile = open(imgpath, "rb")
                                break
                            except FileNotFoundError:
                                print("couldn't load from " + imgpath)
                                continue

                # Try to load it
                imgdata = None
                if imgfile:
                    imgdata = get_gif_pixels(imgfile)
                    imgfile.close()
                else:
                    print("texture " + mat.name + " was not loaded")

                if imgdata:
                    image_data = [comp for rgba in imgdata.pixels for comp in rgba]

                    image = bpy.data.images.new(mat.name, imgdata.width, imgdata.height, alpha=True)

                    from array import array
                    image.pixels = array('f', image_data)
                    image.pack()
                    
                    tex_node.image = image
                else:
                    print("texture was not loaded")
                    
            elif mat.type == MAT_COLOR: 
                shader_node.inputs[0].default_value = (mat.red / 255.0, mat.green / 255.0, mat.blue / 255.0, 1.0)

            if(m.version == 4):
                bmat.transp = mat.transparency
                bmat.illum = mat.self_illum

        else:
            bmat = bpy.data.materials[mat.name]

    ##### Blenderize Objects #####
    for o in m.subobjects:
        palette_mat = None
        o.points = [p for idx, p in enumerate(m.points) if idx >= o.point_start and idx < o.point_start + o.point_num]

        new_obj = bpy.data.objects.new(o.name, bpy.data.meshes.new(o.name))
        new_obj.data.from_pydata(o.points, [], [p.points for p in o.polys])
        o.instance = new_obj

        for p, bp in zip(o.polys, new_obj.data.polygons):
            ## Assign materials 
            if not (p.type & 0x20): # if this isn't a palette color
                if m.materials[p.mat_id].name not in new_obj.data.materials:
                    new_obj.data.materials.append(bpy.data.materials[m.materials[p.mat_id].name])
                bp.material_index = new_obj.data.materials.find(m.materials[p.mat_id].name)
            else: 
                # palette color. TODO do this properly, by pulling from actual palette
                if not palette_mat:
                    palette_mat = bpy.data.materials.new("palette_mat")
                    palette_mat.use_nodes = True

                    nodes = palette_mat.node_tree.nodes
                    nodes.remove(nodes.get('Principled BSDF'))
                    shader_node = nodes.new(type='ShaderNodeBsdfDiffuse')
                    output_node = nodes.get('Material Output')
                    shader_node.inputs[0].default_value = (1.0, 0.0, 1.0, 1.0)
                    links = nodes.data.links
                    links.new(shader_node.outputs[0], output_node.inputs[0])

                    new_obj.data.materials.append(bpy.data.materials["palette_mat"])
                    
                bp.material_index = new_obj.data.materials.find("palette_mat")

        uv_data = [uv for p in o.polys for uv in p.uvs]
        uv_map = new_obj.data.uv_layers.new(do_init=False)
        if uv_data is not None:
            for loop, uv in zip(uv_map.data, uv_data):
                loop.uv = uv

        ## Vhots
        if use_empties:
            for vhot in o.vhots:
                bpy.ops.object.empty_add(type="PLAIN_AXES", location=vhot.offset)
                bpy.context.object.name = "vhot" + str(vhot.id)
                bpy.context.object.parent = new_obj
                bpy.context.object.show_in_front = True
                bpy.context.object.show_name = True

        new_obj.data.flip_normals()
        new_obj.data.validate(verbose=True)
        bpy.data.collections[0].objects.link(new_obj)

    for o in m.subobjects:

        if use_empties:
            if o.type == "rotation": 
                bpy.ops.object.empty_add(type="SINGLE_ARROW", rotation=(0,PI/2.0,0), location=(0,0,0))
                bpy.context.object.parent = o.instance
                bpy.context.object.name = "rot" + str(o.parm)
                bpy.context.object.show_in_front = True
                bpy.context.object.show_name = True
            elif o.type == "translation": 
                bpy.ops.object.empty_add(type="SINGLE_ARROW", rotation=(0,PI/2.0,0), location=(0,0,0))
                bpy.context.object.parent = o.instance
                bpy.context.object.name = "trans" + str(o.parm)
                bpy.context.object.show_name = True

        if o.parent:
            o.transform.transpose()
            o.instance.parent = o.parent.instance
            o.instance.matrix_world = o.transform


    ## bounding box
    if use_empties:
        main_obj = m.subobjects[0]
        bpy.ops.object.empty_add(type="CUBE")
        bpy.context.object.parent = main_obj.instance
        bpy.context.object.name = "bbox"
        # for reasons I don't understand, the scale param in empty_add doesn't work, so:
        bpy.context.object.scale = (m.bounds_max.x, m.bounds_max.y, m.bounds_max.z) # TODO adjust by bounds min

        bpy.ops.object.empty_add(type="SPHERE")
        bpy.context.object.parent = main_obj.instance
        bpy.context.object.name = "bsphere"
        bpy.context.object.scale = (m.bounding_radius, m.bounding_radius, m.bounding_radius)
        bpy.context.object.hide_set(True) # not visible by default

    return {'FINISHED'}


######## UTILITIES ########

# returns floating point list of (r,g,b,a) tuples, or empty list on error
# * assumes sane raster data, and performs few checks
# * does not deinterlace
# * cannot have any extension blocks
# * Outputs 1.0 alpha for every pixel; no transparency (FIXME)
# * will only return the first frame of an animated gif
# * Everything will be upside down, flipped on Y axis (BUG)
def get_gif_pixels(f):
    # just classes to cram values
    class ImgData: pass
    class GIFHeader: pass
    class GIFImage: pass

    header = GIFHeader()
    palette = []
    (header.tag, header.width, header.height, header.flags,
     header.transparent_index, header.pixel_aspect) = unpack("< 6s 2H 3B", f.read(13))
    if header.tag not in (b'GIF87a', b'GIF89a'):
        print("Not a valid GIF")
        return []

    color_table_size = 1 << ((header.flags & 0x07) + 1)
    color_table_depth = (header.flags & 0x70) + 1 # will this ever matter? It might.
    color_table_present = bool(header.flags & 0x80)
    if(color_table_present):
        for i in range(color_table_size):
            r,g,b = unpack("<3B",f.read(3))
            if i == header.transparent_index:
                palette.append((0,0,0,0))
            else:
                palette.append((float(r/255.0), float(g/255.0), float(b/255.0), float(1.0)))

    # TODO should at least allow that one fixed-size extension block
    # but this might never matter
    sentinel = unpack("<B", f.read(1))[0]
    if sentinel != 0x2C:
        print("Image Descriptor must immediately follow GIF header")
        return []

    img = GIFImage()
    img.left, img.top, img.width, img.height, img.flags = unpack("<4H B", f.read(9))
    img.color_table_present = bool(img.flags & 1)
    img.color_table_size = 1<<((img.flags&0x07)+1)
    img.interlaced = bool(img.flags & 2)
    img.color_table_depth = ((img.flags & 0x70) >> 4) + 1

    if(img.color_table_present):
        palette = []
        for i in range(img.color_table_size):
            r,g,b = unpack("<3B",f.read(3))
            if i == header.transparent_index:
                palette.append((0,0,0,0))
            else:
                palette.append((float(r/255.0), float(g/255.0), float(b/255.0), float(1.0)))

    if palette == []:
        print("No color table present")
        return []

    def clear_table(base_width):
        table = [bytes((i,)) for i in range((1 << (base_width)))]
        table.append('CLEAR')
        table.append('END')
        return table

    def get_code(block, bit_cursor, code_width):
        byte_cursor = int(bit_cursor / 8)
        offset_into_byte = int(bit_cursor % 8)
        b1 = block[byte_cursor]
        b2 = block[byte_cursor + 1]
        b3 = int(0)
        if offset_into_byte + code_width > 16:
            b3 = block[byte_cursor + 2]
        mask = ((1<<code_width) - 1) << offset_into_byte
        code = ((b1 | (b2<<8) | (b3<<16)) & mask) >> offset_into_byte
        return code

    base_width = unpack("<1B", f.read(1))[0]
    code_width = base_width + 1

    block = ()
    while True:
        num_bytes = unpack("<1B", f.read(1))[0]
        if num_bytes != 0:
            block += unpack("<"+str(num_bytes) + "B", f.read(num_bytes))
        else:
            break

    table = clear_table(base_width)
    clear_code = 1<<base_width
    end_code = clear_code + 1

    bit_cursor = 0
    prev_code = get_code(block, bit_cursor, code_width)
    if prev_code == clear_code:
        bit_cursor += code_width
        prev_code = get_code(block, bit_cursor, code_width)
    prev_output = table[prev_code]
    result = table[prev_code]

    bit_cursor += code_width
    next_output = None
    while True:
        next_code = get_code(block, bit_cursor, code_width)

        if next_code < len(table):
            if next_code == clear_code:
                table = clear_table(base_width)
                bit_cursor += code_width
                code_width = base_width + 1
                prev_code = get_code(block, bit_cursor, code_width)
                prev_output = table[prev_code]
                result += table[prev_code]
                bit_cursor += code_width
                continue
            elif next_code == end_code:
                break
            else:
                next_output = table[next_code]
                table.append(prev_output + next_output[:1])
        else:
            next_output = table[prev_code] + table[prev_code][:1]
            table.append(next_output)

        result += next_output

        prev_code = next_code
        prev_output = next_output

        bit_cursor += code_width

        if len(table) == 1<<code_width:
            if code_width < 12:
                code_width += 1

    pixels = []
    for i in result:
        pixels.append(palette[i])

    imgdata = ImgData()
    imgdata.pixels = pixels
    imgdata.width = img.width
    imgdata.height = img.height
    return imgdata

from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.props import StringProperty, FloatProperty, BoolProperty, EnumProperty, IntProperty
from bpy.types import Operator



class DarkMaterialProperties(bpy.types.Panel):
    bl_idname = 'DE_MATPANEL_PT_dark_engine_exporter'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'material'
    bl_label = 'Dark Engine Materials (NewDark Toolkit)'

    def draw(self, context):
        activeMat = context.active_object.active_material
        layout = self.layout
        layout.row().prop(activeMat, 'transp')
        layout.row().prop(activeMat, 'illum')


class ImportBin(Operator, ImportHelper):
    bl_idname = "import_scene.binfile"
    bl_label = "Import Bin File"

    filename_ext = ".bin"
    filter_glob: StringProperty(
        default="*.bin",
        options={'HIDDEN'},
        maxlen=255,
    )

    do_search: BoolProperty(
        name="Check common dirs for textures",
        description="Searches the bin's directory and the sub-directories /txt and /txt16",
        default=True,
    )

    use_empties: BoolProperty(
        name="Use empties",
        description="Vhots and joint pivots will be represented as empties in the scene-graph",
        default=False,
    )

    def execute(self, context):
        class ImportOptions: pass
        options = ImportOptions()
        options.do_search = self.do_search
        options.use_empties = self.use_empties
        return import_bin(context, self.filepath, self.do_search, self.use_empties)


def menu_func_import(self, context):
    self.layout.operator(ImportBin.bl_idname, text="Dark bin")

def register():
    bpy.utils.register_class(ImportBin)

    bpy.types.Material.transp = FloatProperty(name='Transparency', description='How transparent this material is. 0.0 = opaque (default), 1.0 = transparent', min=0.0, max=1.0)
    bpy.types.Material.illum = FloatProperty(name='Illumination', description='How emissive the material is. 0 = use natural lighting (default), 1.0 = fully illuminated', min=0.0, max=1.0)
    bpy.utils.register_class(DarkMaterialProperties)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.utils.unregister_class(ImportBin)
    bpy.utils.unregister_class(DarkMaterialProperties)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    register()
