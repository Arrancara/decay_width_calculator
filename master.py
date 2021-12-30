# -*- coding: utf-8 -*-
"""
PHYS20161 Final Assignment: Z0 Boson

This python script reads in data from two csv files. It then merges, and
validates the two data files. The program filters data points and removes
outliers. The data files are also checked. The script fits the dataset to two
fitting parameters and outputs the fits as well as the uncertainities
in the fit. Several plots are also displayed.
Aavash Subedi:  12/12/2021
"""

import sys
from time import sleep
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants

FILE_LOC_1 = "z_boson_data_1.csv"
FILE_LOC_2 = "z_boson_data_2.csv"
INITIAL_GUESS = [90, 3]

KNOWN_WIDTHS = {"ee": 0.08391,
                "muons": 0.08399,
                "tau": 0.08407,
                "invisible": 0.4990,
                "hadrons": 1.744}

def input_validation(accepted_input, user_input):
    """
    Validates users input by checking if it is present within a list of
    approved inputs.

    Parameters
    ----------
    accepted_input : array
        List of accepted inputs.
    user_input : string

    Returns
    -------
    user_input: string
        Validated input is outputted.
    """
    while ("".join(user_input.split())).lower() not in accepted_input:
        user_input = input("Please enter a valid input: ")

    return user_input.lower()

def get_decay_products(accepted_input, product=None):
    """
    Gets the desired decay product from the user. Fed into other functions.

    Parameters
    ----------
    accepted_input : string
        User input.
    product : string, optional
        Gets the key of the decay products. The default is None.

    Returns
    -------
    product : string
        Decay product of the experiment.

    """
    while not product:

        print("Here is the accepted list of inputs: ")
        for elements in accepted_input:
            print(elements)
        product = input("Please enter a value from the accepted list of "
                        "inputs: ")
        product = input_validation(accepted_input.keys(), product)
    return product

def yes_no_checker(user_input):
    """
    Function for checking the users' input is either yes or no. If the input
    is neither the user is prompted to enter a valid response. Outputs True or
    False depending on the response.
    Parameters
    ----------
        user_input : string
        Input obtained from the user.

    Returns
    -------
    Boolean
        Returns True or False depending on users' input.

    """
    checked_response = input_validation(["yes", "no"], user_input)
    return checked_response == "yes"

def cross_section_function(energy_values, mass_z, partial_width_z,
                           product_partial_width=
                           KNOWN_WIDTHS[get_decay_products(KNOWN_WIDTHS)]):
    """
    Returns the cross secton for the Z_0 boson, calcualting its values from the
    given parameters. An optional parameter of PARTIAL_WIDTH_EE  added which
    is used in the calculation. It is left as optional as to allow the code to
    be modified in the future so that the partial width for Z_0 boson's decay
    into electrons.
    Parameters
    ----------

    energy_values: float
    mass_z: float
    partial_width_z: float
        Partial width of boson.

    Optional Parameters
    -------------------

    partial_width_product: float
        Obtains the partial width for a given key by first obtaining the key
        of the product from the user then finidng the the width from a
        dictionary.

    Returns
    ----------
    cross_section: float

    """
    numerator = (12 * np.pi) * energy_values ** 2 * (product_partial_width)**2
    denominator = (mass_z ** 2) * ((energy_values ** 2 - mass_z ** 2) ** 2 +
                                   (mass_z ** 2) * (partial_width_z) ** 2)
    cross_section = (numerator / denominator) * 389400
    return cross_section

def chi_square(prediction, observation, observation_uncertainty):
    """
    Returns the chi squared.

    Parameters
    ----------
    observation : numpy array of floats
    observation_uncertainty : numpy array of floats
    prediction : numpy array of floats
    Returns
    -------
    float
    """

    return np.sum((observation - prediction)**2 / observation_uncertainty**2)


def check_for_file(file_path, datafound=False):
    """
    Checks to see if the file exits. Returns a boolean value depending on
    the existence of the file.

    Parameters
    ----------
    file_path : string
        Location of the file.
    datafound : Boolean, optional
        Existence of datafiles. The default is False.

    Returns
    -------
    DATAFOUND : Boolean

    """
    try:
        file_checker = open(file_path, 'r')
        datafound = True
        file_checker.close()
        return datafound
    except FileNotFoundError:
        print("File not found")
        return datafound

def two_dimensional_plot(input_array, labels, predicted_data):
    """
    Function that takes in parameters and produces a two dimensional plot
    using matplotlib. The plots are stored by their title as a png file.

    Parameters
    ----------
    input_array : array
        Contains all parameters necessary for the plot, x and y data as well
        as uncertainities.
    labels : array
        Contains all labels for the data.
    predicted_data : TYPE
        Data that is obtained from prediction using the model.

    Returns
    -------
    None.

    """

    plot_title, x_label, y_label = labels
    x_data = input_array[:, 0]
    y_data = input_array[:, 1]
    y_uncertainties = input_array[:, 2]
    figure = plt.figure(figsize=(8, 6))
    axes = figure.add_subplot(111)
    axes.errorbar(x_data, y_data, yerr=y_uncertainties,
                  linestyle="None", c="k", fmt='x')
    axes.scatter(x_data, y_data, s=22, c="black", alpha=0.4,
                 label="Data points")


    axes.grid()
    axes.plot(x_data, predicted_data, label="Fitted line", c="r")
    axes.legend()
    axes.set_xlabel(x_label, fontsize=10)
    axes.set_ylabel(y_label, fontsize=10)
    axes.set_title(plot_title, fontsize=20)
    axes.set_ylim(ymin=0)
    plt.savefig(str(plot_title)+".png", dpi=800)
    figure.show()

class DataBase():
    """
    Class that stores the array containing the files and manipulates it.
    Allows for different calculations to take place, including importing,
    validating and cleaning files. Performing the fit on the contents of the
    class. Full details is described in the docstring for each function.
    """
    def __init__(self, array=None, fitted_mass=None,
                 fitted_width_boson=None, fitting_parameters=2):
        """
        Initialises the class, stores the contents of the
        file in an array, the fitting parameters and the number of fitting
        parameters used.
        Parameters
        ----------

        array : numpy array, optional
            DESCRIPTION. The default is None.
        fitted_mass: float, optional
            Mass of the boson obtained after fitting. The default is None.
        fitted_width_boson: float, opotional
            Partial width of the boson obtained from fitting. The default is
            None.
        fitting_parameters: float,optional
            Number of fitting paramets used. Is changed if the user wants to
            fit for the partial width of electrons. The default is 2.
        Returns
        -------
        None.

        """
        self.fitting_parameters = fitting_parameters
        self.array = array
        self.fitted_mass = fitted_mass
        self.fitted_width_boson = fitted_width_boson

    def file_validater(self):
        """
        Checks if the file being passed is valid by checking if it has an
        appropriate number of columns and contains rows. Retruns boolean
        values depending on check.
        """
        if np.shape(self.array)[0] == 0:
            print("Missing rows, please check again.\n"
                  "The script will close in 5 seconds.")
            sleep(5)
            sys.exit()
        if np.shape(self.array)[1] < 3:
            print("Missing rows, please check again.\n"
                  "The script will close in 5 seconds.")
            sleep(5)
            sys.exit()
    def parse_data(self, file_path, delimiter_used=",", comments_used=None,
                   skip_header_set=None):
        """
        Parses the data if the file is found and stores in class.array as a
        numpy array. Takes optional arguments that are preset to certain
        values. E.g. The delimeter is set to comma as csv files is
        generally used, however the user can input a custm delimeter which
        would override the comma.

        Parameters
        ----------
        file_path : TYPE
            Path of the file.
        delimiter_used : Boolean, optional
            Delimeter to pass to the function. The default is ",".
        comments_used : Boolean, optional
            Comments specifier, numpy will ignore any rows staring with the
            comment. The default is None.
        skip_header_set : float, optional
            The number of rows numpy should skip whilst parsing the file.
            The default is None.

        Returns
        -------
        None.

        """
        if check_for_file(file_path):
            self.array = np.genfromtxt(file_path, delimiter=delimiter_used,
                                       comments=comments_used,
                                       skip_header=skip_header_set,
                                       filling_values="nan")
        else:
            print("Please check file directory and run the script again.\n"
                  "The script will close in 5 seconds.")
            sleep(5)
            sys.exit()
        self.file_validater()

    def remove_corrupt_data(self):
        """
        Removes rows containing data points that either have a value of "nan"
        or a negative value which is unphysical. The parsing function ensures
        that a filling value of nan is added to corrupt data points.

        Returns
        -------
        None.

        """

        self.array = self.array[~np.isnan(self.array).any(axis=1)]
        self.array = self.array[~np.any(self.array <= 0, axis=1)]

    def add_data(self, other):
        """
        Adds the stored array of another object of class database to the
        original object.
        Parameters
        ----------
        other : database class
            An object belonging to the custom class, "database".

        Returns
        -------
        None.

        """
        self.array = np.vstack((self.array, other))

    def sort_data(self):
        """
        Sorts the array within the class by increasing value by elements in
        the first row. In this case, the first row contains the energy values.

        Returns
        -------
        None.

        """
        self.array = self.array[self.array[:, 0].argsort()]
    def number_of_sigmas(self):
        """
        Calcualtes the number of sigmas to classify as an outlier, based on a
        stastical method. Checks in the number of rows (data points) to
        calculate the number of sigmas. Any points outside these sigmas
        is classified as an outlier.
        Returns
        -------
        float.
            Number of standard deviation to reject data points as
            uncertainties.
        """
        if np.shape(self.array)[0] > 1500000:
            return 5
        if np.shape(self.array)[0] > 330:
            return 4
        if np.shape(self.array)[0] > 20:
            return 3
        return 2

    def fit_data(self, initial_values=np.array(INITIAL_GUESS)):
        """
        Fits the data using curve fit to the cross section function

        Parameters
        ----------
        initial_values : array, optional
            The original values that are fed into the fitting model.
            The default is none.

        Returns
        -------
        popt : Array
            Contains the fitted values.
        pcov : Array
            Covariance matrix of the fitted values.


        """
        if self.fitting_parameters == 3:
            initial_values = np.append(initial_values, KNOWN_WIDTHS["ee"])
        try:
            popt, pcov = curve_fit(cross_section_function, self.array[:, 0],
                                   self.array[:, 1], sigma=self.array[:, 2],
                                   p0=initial_values)
            setattr(self, 'fitted_mass', popt[0])
            setattr(self, "fitted_width_boson", popt[1])
        except RuntimeError:
            print("The curve fitting failed with the initial guess, please"
                  "change the initial guess in the function.\n"
                  "The script will close in 5 seconds.")
            sleep(5)
            sys.exit()
        return popt, pcov


    def strong_outlier_removeal(self):
        """
        Removes the strong outliers contained within the objects' array.
        A strong outlier is classified as any values that lies outside 3
        interquartile ranges from the lower and upper quartiles.

        Returns
        -------
        None.

        """

        lower_quartile = np.quantile(self.array[:, 1], 0.25,
                                     interpolation="midpoint")


        upper_quartile = np.quantile(self.array[:, 1], 0.75,
                                     interpolation="midpoint")
        interquartile_range = upper_quartile-lower_quartile
        self.array = self.array[~(self.array[:, 1]
                                  > upper_quartile + 3*interquartile_range)]
        self.array = self.array[~(self.array[:, 1]
                                  < lower_quartile - 3*interquartile_range)]

    def outlier_removeal(self):
        """
        Removes the outliers that are outside a certain number of standard
        deviations from the fitted value.

        Parameters
        ----------
        None.
        Returns
        -------
        None.

        """
        self.fit_data()
        sigmas = self.number_of_sigmas()


        if self.fitting_parameters == 3:
            self.array = self.array[np.abs(np.abs((
                cross_section_function(self.array[:, 0], self.fitted_mass,
                                       self.fitted_width_boson,
                                       self.fit_data()[0][2])
                - self.array[:, 1])) < sigmas*self.array[:, 2])]

        else:
            self.array = self.array[np.abs(np.abs((cross_section_function(
                self.array[:, 0], self.fitted_mass, self.fitted_width_boson)
                                                   - self.array[:, 1]))
                                           < sigmas*self.array[:, 2])]

    def data_cleaner(self, initial_outlier_removed=False):
        """
        Groups together previous functions, that is used to remove all
        outliers and corrupt data from the object and sort the data.
        Uses recursion to first remove the strong outliers and corrupt data,
        then proceeds to remove the outliers obtained after the fitting.

        Parameters
        ----------
        initial_outlier_removed : Boolean, optional
            Checks if the function has already run before.
            The default is False.

        Returns
        -------
        Function
            Function recurses over itself to firstly remove strong outliers
            then remove outliers that lie within a certain number of standard
            deviations away.
        Int
            Filler return to show that data has been cleaned.

        """


        if not initial_outlier_removed:
            self.remove_corrupt_data()
            self.sort_data()
            self.strong_outlier_removeal()
            initial_outlier_removed = True
            return self.data_cleaner(self)
        self.outlier_removeal()
        return 1

    def calculate_lifetime(self):
        """
        Calculates the lifetime of the particle before it decays.

        Returns
        -------
        Float
        Lifetime value of the boson obtained from
        """
        return constants.hbar/(constants.elementary_charge*10**(9)*(
            float(self.fitted_width_boson)))

    def reduced_chi_square_calculator(self, mass_value=None,
                                      width_value_boson=None,
                                      width_value_ee=None):
        """
        Calculates the reduced chi-squared for the set of data points, using
        the fitting model.
        Parameters
        ----------
        mass_value : TYPE, optional
            Mass of the z_boson. The default is None.
        width_value : TYPE, optional
            Partial width of Z-boson. The default is None.

        Returns
        -------
        float
            Reduced chi-squared given initial inputs.

        """
        try:
            if not mass_value:
                mass_value = self.fitted_mass
        except ValueError:
            pass
        try:
            if not width_value_boson:
                width_value_boson = self.fitted_width_boson
        except ValueError:
            pass

        if self.fitting_parameters == 3:
            try:
                if not width_value_ee:
                    width_value_ee = self.fit_data()[0][2]
            except ValueError:
                pass
            return chi_square(
                cross_section_function(self.array[:, 0],
                                       mass_value, width_value_boson,
                                       width_value_ee),
                self.array[:, 1], self.array[:, 2])/(np.shape(self.array)[0]
                                                     - self.fitting_parameters)

        return chi_square(
            cross_section_function(self.array[:, 0], mass_value,
                                   width_value_boson),
            self.array[:, 1], self.array[:, 2]) / (np.shape(self.array)[0] -
                                                   self.fitting_parameters)
    def reduced_chi_squared_mesh(self, number_of_points=None):
        """
        Parameters
        ----------
        number_of_points : TYPE, optional
            Number of points that are evaluated. The default is None.

        Returns
        -------
        data_store : array
            Array with the calculated reduced chi-squared.
        x_mesh: array
        y_mesh: array
        z_mesh: array

        """
        number_of_points = np.shape(self.array)[0]

        x_values = np.linspace(self.fitted_mass-0.8, self.fitted_mass+0.8,
                               number_of_points)
        y_values = np.linspace(self.fitted_width_boson-0.4,
                               self.fitted_width_boson+0.4,
                               number_of_points)

        data_store = np.empty([number_of_points, number_of_points])

        for index_x in range(number_of_points):
            for index_y in range(number_of_points):
                data_store[index_y][index_x] = \
                self.reduced_chi_square_calculator(x_values[index_x],
                                                   y_values[index_y])
        x_mesh, y_mesh = np.meshgrid(x_values, y_values)
        return x_mesh, y_mesh, data_store
    def contour_plot_two_dimensions(self):
        """
        Plots the 2 Dimensional contour plot for a 2 dimensional fit. Saves
        the file as png.

        Returns
        -------
        None.

        """
        data = (self.reduced_chi_squared_mesh())
        fig = plt.figure()
        ax = fig.add_subplot(111)
        contour = ax.contour(data[0], data[1], data[2], 25, cmap="plasma")
        ax.set_xlabel("$Mass$ [$GeVc^{-2}$]", fontsize=10)
        ax.set_ylabel("Boson Width $[GeV]$", fontsize=10)
        ax.set_title(r"$\chi^{2}_{red}$ contours", fontsize=20)
        ax.plot(self.fitted_mass, self.fitted_width_boson, "x", c="k")
        ax.annotate("{0:.3},{1:.3}".format(self.fitted_mass,
                                           self.fitted_width_boson),
                    (self.fitted_mass, self.fitted_width_boson),
                    fontsize=8)
        ax.clabel(contour, fontsize=8)
        plt.savefig("reduced_chi_squared_2D.png", dpi=800)
        fig.show()
    def surface_plot(self):
        """
        Produces a 3D surface plot for a 2-dimensional fit. Saves output as
        a png file.

        Returns
        -------
        None.

        """
        data = self.reduced_chi_squared_mesh()
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        surf = ax.plot_surface(data[0], data[1], data[2], cmap="plasma")
        ax.set_xlabel("$Mass$ [$GeVc^{-2}$]", fontsize=8)
        ax.set_ylabel("Boson Width $[GeV]$", fontsize=8)
        ax.set_title(r"$\chi^{2}_{red}$ vs parameters",
                     fontsize=15)
        ax.set_zlabel(r"$\chi^{2}_{red}$", fontsize=10)
        plt.tight_layout()
        fig.colorbar(surf)
        plt.savefig("reduced_chi_squared_3D.png", dpi=800)
        fig.show()
    def plot_data(self):
        """
        Calculates the predicted data from the fitted parameters, diffrent
        values are depending on the number of fitting parameters. Uses a
        2-D plotting function as described before to produce a plot.

        Returns
        -------
        None.

        """
        if self.fitting_parameters == 3:
            predicted_data = cross_section_function(self.array[:, 0],
                                                    self.fitted_mass,
                                                    self.fitted_width_boson,
                                                    self.fit_data()[0][2])
        else:
            predicted_data = cross_section_function(self.array[:, 0],
                                                    self.fitted_mass,
                                                    self.fitted_width_boson,
                                                    KNOWN_WIDTHS["ee"])
        two_dimensional_plot(self.array, ("Cross section vs Energy",
                                          "$Energy$ [$GeV$]",
                                          "Cross Section [$nb]$"),
                             predicted_data)
def main():
    """
    Main part of the code.

    Returns
    -------
    None.

    """
    file = DataBase()
    file.parse_data(FILE_LOC_1, ",", "%", 1)
    file_2 = DataBase()
    file_2.parse_data(FILE_LOC_2, ",", "%", 1)
    file.add_data(file_2.array)

    number_of_parameters = yes_no_checker(input("Please enter yes or no.\n"
                                                "Would you like to fit "
                                                "for a third parameter? "))
    if number_of_parameters:
        file.fitting_parameters = 3
    file.data_cleaner()
    fit_results, fit_uncertainties = file.fit_data()
    file.plot_data()
    if file.fitting_parameters == 3:
        print("The fitted width of EE is {0:.4}"
              " +/- {1:.2} GeV.".format(
                  fit_results[2], sqrt(fit_uncertainties[2][2])))
    else:
        file.contour_plot_two_dimensions()
        contour_plot_3 = yes_no_checker(input("Please enter yes or no.\n"
                                              "Would you like to plot a "
                                              "3D plot?\n"
                                              "Please note this is time "
                                              "and resource intensive: "))
        if contour_plot_3:
            file.surface_plot()

    print("The fitted mass is {0:.4} +/- {1:.2}"
          " GeVc^-2.".format((fit_results[0]),
                             sqrt(fit_uncertainties[0][0])))
    print("The fitted width of the decay product is {0:.4} +/- {1:.2}"
          " GeV.".format(fit_results[1],
                         sqrt(fit_uncertainties[1][1])))
    print("The reduced chi-squared of the fitting"
          " is {0:.3f}.".format(float(file.reduced_chi_square_calculator())))
    time_uncertainty = (float(file.calculate_lifetime())
                        *sqrt(fit_uncertainties[1][1])/fit_results[1])
    print("The decay time is {0:.3} +/- {1:.2} seconds."
          .format(float(file.calculate_lifetime()), time_uncertainty))

if __name__ == "__main__":
    main()
