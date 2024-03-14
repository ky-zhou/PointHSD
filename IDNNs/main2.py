"""
Train % plot networks in the information plane
"""
from idnns.networks import information_network2 as inet
def main():
    #Build the network
    print ('Building the network')
    net = inet.informationNetwork()
    net.print_information()
    # print ('Start laoding the data')
    # net.load_data()
    print ('Start calculating')
    net.run_calc_only()
    print ('Saving data')
    net.save_data_calc_only()
    print ('Ploting figures')
    #Plot the newtork
    net.plot_network_calc_only()
if __name__ == '__main__':
    main()

