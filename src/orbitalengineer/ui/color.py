import cairo

def hex_to_rgba(hexcolor:str) -> tuple:
    hexcolor = hexcolor.lstrip("#")    
    if len(hexcolor) == 6:
        r, g, b = (
            int(hexcolor[0:2], 16),
            int(hexcolor[2:4], 16),
            int(hexcolor[4:6], 16),
        )
        a = 255
    elif len(hexcolor) == 8:
        r, g, b, a = (
            int(hexcolor[0:2], 16),
            int(hexcolor[2:4], 16),
            int(hexcolor[4:6], 16),
            int(hexcolor[6:8], 16),
        )
    else:
        raise Exception("Invalid color value. Must be of RRGGBB or RRGGBBAA")
    return ( r/255.0, g/255.0, b/255.0, a/255.0 )

def set_source_hex(cr:cairo.Context, hexcolor:str):
    r,g,b,a = hex_to_rgba(hexcolor)
    cr.set_source_rgba(r, g, b, a)