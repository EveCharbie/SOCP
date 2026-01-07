import matplotlib

matplotlib.use("TkAgg")  # or 'Qt5Agg'
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
from casadi import Function, Callback, nlpsol_out, nlpsol_n_out, Sparsity
import multiprocessing as mp


class OnlineCallback(Callback):
    """
    CasADi interface of Ipopt callbacks
    """

    def __init__(self, nx: int, ng: int, grad_f_func: Function, grad_g_func: Function, g_names: list[str], ocp):
        """
        Parameters
        ----------
        nx: int
            The number of optimization variables
        ng: int
            The number of constraints
        """
        Callback.__init__(self)
        self.nx = nx
        self.ng = ng
        self.grad_f_func = grad_f_func
        self.grad_g_func = grad_g_func
        self.g_names = g_names
        self.time_vector = np.linspace(1, ocp["n_shooting"], ocp["n_shooting"] + 1)

        # Create the ipopt output plot
        self.construct("plots", {})

        self.queue = mp.Queue()
        self.plotter = ProcessPlotter(self)
        self.plot_process = mp.Process(
            target=self.plotter, args=(self.queue, {"lbw": ocp["lbw"], "ubw": ocp["ubw"]}), daemon=True
        )
        self.plot_process.start()

    def close(self):
        self.plot_process.kill()

    @staticmethod
    def get_n_in() -> int:
        """
        Get the number of variables in

        Returns
        -------
        The number of variables in
        """

        return nlpsol_n_out()

    @staticmethod
    def get_n_out() -> int:
        """
        Get the number of variables out

        Returns
        -------
        The number of variables out
        """

        return 1

    @staticmethod
    def get_name_in(i: int) -> int:
        """
        Get the name of a variable

        Parameters
        ----------
        i: int
            The index of the variable

        Returns
        -------
        The name of the variable
        """

        return nlpsol_out(i)

    @staticmethod
    def get_name_out(_) -> str:
        """
        Get the name of the output variable

        Returns
        -------
        The name of the output variable
        """

        return "ret"

    def get_sparsity_in(self, i: int) -> tuple:
        """
        Get the sparsity of a specific variable

        Parameters
        ----------
        i: int
            The index of the variable

        Returns
        -------
        The sparsity of the variable
        """

        n = nlpsol_out(i)
        if n == "f":
            return Sparsity.scalar()
        elif n in ("x", "lam_x"):
            return Sparsity.dense(self.nx)
        elif n in ("g", "lam_g"):
            return Sparsity.dense(self.ng)
        else:
            return Sparsity(0, 0)

    def create_ipopt_output_plot(self):
        """
        This function creates the plots for the ipopt output: f, g, inf_pr, inf_du.
        """
        self.f_sol = []
        self.inf_pr_sol = []
        self.inf_du_sol = []
        self.grad_f_sol = []
        self.grad_g_sol = []
        self.lam_x_sol = []
        self.unique_g_names = []
        for name in self.g_names:
            if name not in self.unique_g_names:
                self.unique_g_names += [name]
        self.g_sol = {name: [] for name in self.unique_g_names}

        ipopt_fig, axs = plt.subplots(4, 1, num="IPOPT output")
        axs[0].set_ylabel("f", fontweight="bold")
        axs[1].set_ylabel("constraints", fontweight="bold")
        axs[2].set_ylabel("inf_pr", fontweight="bold")
        axs[3].set_ylabel("inf_du", fontweight="bold")

        plots = []
        colors = get_cmap("viridis")
        for i in [0, 2, 3]:
            plot = axs[i].plot([0], [1], linestyle="-", marker=".", color="k")
            plots.append(plot[0])
            axs[i].grid(True)
            axs[i].set_yscale("log")

        plot = axs[3].plot([0], [1], linestyle="-", marker=".", color=colors(0.1), label="grad_f")
        plots.append(plot[0])
        plot = axs[3].plot([0], [1], linestyle="-", marker=".", color=colors(0.5), label="grad_g")
        plots.append(plot[0])
        plot = axs[3].plot([0], [1], linestyle="-", marker=".", color=colors(0.9), label="lam_x")
        plots.append(plot[0])
        axs[3].legend()

        # Add all g plots at the end
        for i_g, name in enumerate(self.unique_g_names):
            plot = axs[1].plot(
                [0], [1], linestyle="-", marker=".", label=name, color=colors(i_g / len(self.unique_g_names))
            )
            plots.append(plot[0])
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_yscale("log")

        self.ipopt_fig = ipopt_fig
        self.ipopt_plots = plots
        self.ipopt_axes = axs

    def get_g_by_name(self, g: np.ndarray):
        g_by_name = {name: [] for name in self.unique_g_names}
        for i_g, name in enumerate(self.g_names):
            g_by_name[name].append(float(g[i_g]))
        g_max_by_name = {name: np.max(np.abs(values)) for name, values in g_by_name.items()}
        return g_max_by_name

    def get_gmin_gmax(self, g_sol):
        g_min = np.inf
        g_max = 0
        for name in self.unique_g_names:
            g_min = min(g_min, np.min(g_sol[name]))
            g_max = max(g_max, np.max(g_sol[name]))
        return g_min, g_max

    def update_ipopt_output_plot(self, args):
        """
        This function updated the plots for the ipopt output: x, f, g, inf_pr, inf_du.
        We currently do not have access to the iteration number, weather we are currently in restoration, the lg(mu), the length of the current step, the alpha_du, or the alpha_pr.
        inf_pr is obtained from the maximum absolute value of the constraints.
        inf_du is obtained from the maximum absolute value of the equation 4a in the ipopt original paper.
        """

        x = args["x"]
        f = args["f"]
        g = args["g"]
        lam_x = args["lam_x"]
        lam_g = args["lam_g"]

        inf_pr = np.max(np.abs(g))

        grad_f = self.grad_f_func(x)
        grad_g_lam = self.grad_g_func(x) @ lam_g
        eq_4a = np.max(np.abs(grad_f + grad_g_lam - lam_x))
        inf_du = np.max(np.abs(eq_4a))

        self.f_sol.append(float(f))
        self.inf_pr_sol.append(float(inf_pr))
        self.inf_du_sol.append(float(inf_du))
        self.grad_f_sol.append(float(np.max(np.abs(grad_f))))
        self.grad_g_sol.append(float(np.max(np.abs(grad_g_lam))))
        self.lam_x_sol.append(float(np.max(np.abs(lam_x))))
        g_max_by_name = self.get_g_by_name(g)
        for name in self.unique_g_names:
            self.g_sol[name].append(g_max_by_name[name])
        g_min, g_max = self.get_gmin_gmax(self.g_sol)

        self.ipopt_plots[0].set_ydata(self.f_sol)
        self.ipopt_plots[1].set_ydata(self.inf_pr_sol)
        self.ipopt_plots[2].set_ydata(self.inf_du_sol)
        self.ipopt_plots[3].set_ydata(self.grad_f_sol)
        self.ipopt_plots[4].set_ydata(self.grad_g_sol)
        self.ipopt_plots[5].set_ydata(self.lam_x_sol)
        for i_g, name in enumerate(self.unique_g_names):
            self.ipopt_plots[6 + i_g].set_ydata(self.g_sol[name])

        self.ipopt_axes[0].set_ylim(np.min(self.f_sol), np.max(self.f_sol))
        self.ipopt_axes[1].set_ylim(1e-10, g_max)
        self.ipopt_axes[2].set_ylim(np.min(self.inf_pr_sol), np.max(self.inf_pr_sol))
        self.ipopt_axes[3].set_ylim(
            # np.min(
            #     np.array(
            #         [
            #             1e8,
            #             np.min(np.abs(self.inf_du_sol)),
            #             np.min(np.abs(self.grad_f_sol)),
            #             np.min(np.abs(self.grad_g_sol)),
            #             np.min(np.abs(self.lam_x_sol)),
            #         ]
            #     )
            # ),
            1e-10,
            np.max(
                np.array(
                    [
                        1e-8,
                        np.max(np.abs(self.inf_du_sol)),
                        np.max(np.abs(self.grad_f_sol)),
                        np.max(np.abs(self.grad_g_sol)),
                        np.max(np.abs(self.lam_x_sol)),
                    ]
                )
            ),
        )

        for i in range(len(self.ipopt_plots)):
            self.ipopt_plots[i].set_xdata(range(len(self.f_sol)))
        for i in range(4):
            self.ipopt_axes[i].set_xlim(0, len(self.f_sol))

    def eval(self, arg: list | tuple) -> list:
        """
        Send the current data to the plotter

        Parameters
        ----------
        arg: list | tuple
            The data to send

        Returns
        -------
        A list of error index
        """
        send = self.queue.put
        args_dict = {}
        for i, s in enumerate(nlpsol_out()):
            args_dict[s] = arg[i]
        send(args_dict)
        return [0]


class ProcessPlotter(object):

    def __init__(self, online_callback):
        """
        Parameters
        ----------
        online_callback: OnlineCallback
            A reference to the online callback to show
        """

        self.online_callback = online_callback

    def __call__(self, pipe: mp.Queue, options: dict):
        """
        Parameters
        ----------
        pipe: mp.Queue
            The multiprocessing queue to evaluate
        options: dict
            The option to pass
        """
        self.pipe = pipe
        self.online_callback.create_ipopt_output_plot()
        timer = self.online_callback.ipopt_fig.canvas.new_timer(interval=100)
        timer.add_callback(self.callback)
        timer.start()
        plt.show()

    def callback(self) -> bool:
        """
        The callback to update the graphs

        Returns
        -------
        True if everything went well
        """

        while not self.pipe.empty():
            args = self.pipe.get()
            self.online_callback.update_ipopt_output_plot(args)

        nb_iter = len(self.online_callback.ipopt_axes[0].lines[0].get_xdata())
        self.online_callback.ipopt_fig.canvas.draw()
        if nb_iter % 1000 == 0:
            self.online_callback.ipopt_fig.savefig(f"ipopt_output_{nb_iter}.png")
        self.online_callback.ipopt_fig.canvas.flush_events()

        return True
