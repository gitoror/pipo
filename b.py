import pyopencl as cl

platforms = cl.get_platforms()
pl = platforms[0]
print(pl.name)
print(pl.vendor)
print(pl.version)

devices = pl.get_devices()
print(devices)
dv = devices[1]
print(dv.max_work_group_size)
print(dv.max_clock_frequency)

ctx = cl.Context(devices=[dv])
# contexte d'execution, objet avec lequel on communique avec la crte graphique
print(ctx)

# autre manière de créer un contexte directement
# ne pas faire interactive = Fasle sinon choisi le mauvais device
ctx = cl.create_some_context(interactive=False)
print(ctx)
