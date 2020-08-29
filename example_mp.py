# This is an example for multiprocessing task
from termcolor import colored
import multiprocessing, time

def nav(server, comfirmed):
    for i in range(10):
        server.send(i)
        print(colored('Server: ','magenta') + 'Sent Data from nav: {}'.format(i))
        while True:  # Waiting for client to confirm
            if comfirmed.value:
                break
        comfirmed.value = 0  # Turn off the switch

    server.send('END')
    print(colored('Server: ','magenta') + 'END')

def ssg(client, comfirmed):
    while True:
        msg = client.recv()
        if msg == 'END':
            break
        time.sleep(1)
        print(colored('Client: ','blue') + 'Receive Data from nav: {}'.format(msg))
        comfirmed.value = 1

    print(colored('Client: ','blue') + 'END')


if __name__ == '__main__':

    comfirmed = multiprocessing.Value('i')  # Int value: 1 for confirm complete task and other process can go on while 0 otherwise
    comfirmed.value = 0
    server, client = multiprocessing.Pipe()  # server send date and client receive data

    navigator_node = multiprocessing.Process(target=nav, args=(server, comfirmed))
    ssg_node = multiprocessing.Process(target=ssg, args=(client, comfirmed))

    navigator_node.start()
    ssg_node.start()
    navigator_node.join()
    ssg_node.join()
