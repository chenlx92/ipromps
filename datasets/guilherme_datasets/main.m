
% Read and plot data from human-robot collaboration
% taskProMP{1}: task 1
% taskProMP{2}: task 2
% taskProMP{3}: task 3
%
% Trajectories of both human and robot are time aligned.
% Each column is one degree of freedom.
% Each row is a time step.
% data{1}(:,1:3)   xyz coordinates of the human wrist recorded via optical
%                  marker traking
% data{1}(:,4:10)  joint angles of the KUKA lightweight arm, starting from
%                  base towards the end effector
% data{1}(:,11:13) xyz coordinates of the robot end-effector

function main()

    clear; clc; close all;
    dbstop if error;


    load('taskProMP.mat');

    number_of_tasks = numel(taskProMP);
    
    for j=1:number_of_tasks 
        plot_data(taskProMP{j});
    end
    
    for j=1:number_of_tasks 
        gen_csv(taskProMP{j}, j);
    end

end


function plot_data(data)
    h1 = figure; axis equal; grid on; hold on; 
    xlabel 'x (m)';
    ylabel 'y (m)'; 
    zlabel 'z (m)'; 
    view([-1 1 1])
    title('Blue: Human wrist. Red: robot hand')

    number_of_training_data = numel(data);
    for k=1:number_of_training_data
        plot3(data{k}(:,1), data{k}(:,2), data{k}(:,3), 'b'  );
        plot3(data{k}(:,11), data{k}(:,12), data{k}(:,13), 'r'  );
    end
end
    
%  created by Longxin to output the csv file that can be read by python
function gen_csv(data, task_idx)
	number_of_training_data = numel(data);
    for k=1:number_of_training_data
        data_temp = [data{k}(:,1), data{k}(:,2), data{k}(:,3), data{k}(:,11), data{k}(:,12), data{k}(:,13)];
        file_name = ['./csv_datasets/', 'task', num2str(task_idx), '_', num2str(k)];
        csvwrite(file_name, data_temp);
    end
end
