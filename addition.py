import pyopencl as cl
import numpy as np

if __name__ == '__main__':
    platforms = cl.get_platforms()
    pl = platforms[0]
    devices = pl.get_devices()
    dv = devices[1]
    # contexte d'execution, objet avec lequel on communique avec la crte graphique
    ctx = cl.Context(devices=[dv])
    # autre manière de créer un contexte directement
    # ne pas faire interactive = Fasle sinon choisi le mauvais device
    # ctx = cl.create_some_context(interactive=False)
    # print(ctx)

    # queue d'execution, objet avec lequel on envoie les commandes à la carte graphique
    queue = cl.CommandQueue(ctx)
    # prog : objet qui réunit les onfctions compilées sur la GPU
    prog = cl.Program(ctx, open("addition.cl").read()).build()
    # kernel : fonction visible depuis le CPU, doit avoir un résultat void
    # ou fonction rendue accessible par le CPU
    # __global : tous les indices sont dispo par tout le monde,on gérera le param depuis le CPU  (envoi/retours)
    # tableaxu => __global obligatoire
    # float *: type pointeur vers un float, ici a
    # const : le param n'est pas modifié par la fonction, optimise le calcul
    # Types : int, float, int *, float * (tableaux de ...), float ** tableaux à 2D (interdit)
    # 0 : dimension de l'espace
    # 100x20x10 : 0, 1, 2
    # addition de vecteurs
    # build : compile le code (vérifier qeu prg bien écrit) et envoie sur le GPU

    # les flottants de la carte graphique sont des float32
    a_np = np.random.rand(10).astype(np.float32)
    b_np = np.random.rand(10).astype(np.float32)
    res_np = np.zeros_like(a_np)
    mf = cl.mem_flags
    # a_g : buffer sur la carte graphique = vue sur la mémoire de la carte graphique
    # mf.READ_WRITE : on peut avoir a_g a gauche et a droite de l'opérateur =
    a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

    # Le calcul a été fait sur le GPU
    prog.add(queue, (10,), None, a_g, b_g, np.float32(1.03))
    # 3eme paramètres permet de faire des sous catégories
    # a_g et b_g sont les paramètres de la fonction add
    # (10,) : 10 threads lancés en parallèle, dans le code .cl
    # ça correspond a int i = get_global_id(0), donc i va de 0 à 9
    # donc ca permet de faire la somme de vecteurs de taille 10
    # Si on avait mit (5, 2) : int  i = get_global_id(0), int j = get_global_id(1)
    # donc i va de 0 à 4 et j de 0 à 1

    # res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, a_g)  # Récupération du résultat
    # on peut aussi faire un enqueue copy dans l'autre sens, ce qui compte
    # c'est l'ordre des arguments (queue, dest, src)
    print(res_np)
    # vérification, faux pour de mauvaises raisons car erreurs d'arrondis
    res_np == a_np + b_np
    # du au fait que les arrondis sont != sur CPU et GPU
    print(np.allclose(res_np, a_np + b_np+np.float32(1.03)))  # vérification, vrai

    prog.difference(queue, (10,), None, a_g, b_g)
    cl.enqueue_copy(queue, res_np, a_g)
    print(res_np)
    print(np.allclose(res_np, a_np - b_np))
