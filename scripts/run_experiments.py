import mcr.causality.scms.examples as ex
from mcr.experiment.run import run_experiment
from mcr.experiment.compile import compile_experiments
import random
import argparse
import os
import logging
import numpy as np
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Create recourse experiments. "
        + "For every configuration a separate folder is created. "
        + "Within every folder a folder for every interation is created."
        + "The savepath specifies the folder in which these folders shall be placed."
    )

    parser.add_argument("scm_name", help=f"one of {ex.scm_dict.keys()}", type=str)
    parser.add_argument("N", help="Number of observations", type=int)
    parser.add_argument(
        "N_recourse",
        help="Number of individuals for whom recourse shall be computed",
        type=int,
    )
    parser.add_argument("gamma", help="gammas for recourse", type=float)
    parser.add_argument(
        "lbd", help="lbd weight for meaningfulness objective", type=float
    )
    parser.add_argument(
        "savepath",
        help="savepath for the experiment folder. either relative to working directory or absolute.",
        type=str,
    )
    parser.add_argument(
        "n_iterations", help="number of runs per configuration", type=int, default=5
    )

    parser.add_argument("--POP_SIZE", help="population size", default=1000, type=int)
    parser.add_argument("--NGEN", help="number generations", default=400, type=int)
    parser.add_argument(
        "--thresh", help="threshs for prediction and recourse", type=float, default=0.5
    )
    parser.add_argument(
        "--n_digits", help="number of decimal points for rounding", type=int, default=2
    )
    parser.add_argument("--logging_level", type=int, default=20)
    parser.add_argument("--predict_individualized", type=bool, default=True)
    parser.add_argument("--assess_robustness", type=bool, default=False)
    parser.add_argument("--nr_refits", type=int, default=5)
    parser.add_argument("--model_type", type=str, default="logreg")
    parser.add_argument("--t_type", type=str, default="all")
    parser.add_argument("--id", type=int, default=None)
    parser.add_argument("--parallelise", type=bool, default=False)
    parser.add_argument(
        "--ignore_np_errs",
        help="whether to ignore all numpy warnings and errors",
        default=True,
        type=bool,
    )

    start_time = time.time()
    args = parser.parse_args()

    # set logging settings
    logging.getLogger().setLevel(args.logging_level)

    if args.ignore_np_errs:
        np.seterr(all="ignore")

    desc = f"N{args.N}_Nrec{args.N_recourse}_gam{args.gamma}_t{args.thresh}_lbd{args.lbd}_nit{args.n_iterations}_NGEN{args.NGEN}_POPS{args.POP_SIZE}"
    savepath_config = args.savepath + desc
    print(savepath_config)

    n_tries = 0
    done = False
    if args.id is None:
        while n_tries < 5 and not done:
            try:
                n_tries += 1
                config_id = random.randint(0, 1024)
                savepath_config = savepath_config + f"_{config_id}"
                print(savepath_config)
                os.mkdir(savepath_config)
                done = True
            except Exception as err:
                logging.warning(
                    "Could not generate folder...{}".format(savepath_config)
                )
    else:
        config_id = args.id
        savepath_config = savepath_config + f"_{config_id}"
        print("savepath_config: {}".format(savepath_config))
        assert os.path.isdir(savepath_config)

    run_experiment(
        args.scm_name,
        args.N,
        args.N_recourse,
        args.gamma,
        args.thresh,
        args.lbd,
        savepath_config,
        NGEN=args.NGEN,
        iterations=args.n_iterations,
        POP_SIZE=args.POP_SIZE,
        rounding_digits=args.n_digits,
        use_scm_pred=False,
        predict_individualized=args.predict_individualized,
        assess_robustness=args.assess_robustness,
        nr_refits_batch0=args.nr_refits,
        model_type=args.model_type,
        t_types=args.t_type,
        parallelisation=args.parallelise,
    )

    compile_experiments(args.savepath, args.scm_name)

    print("FINISHED TIME:", time.time() - start_time)
