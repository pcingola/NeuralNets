package com.pcingola.neunet.kohonen;

import java.awt.Container;
import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

import com.pcingola.neunet.Network;

/**
 * A Simple Kohonen neural network demo
 * 
 * @author pcingola@sinectis.com
 */
public class KohonenNeuNetDemo implements ActionListener {

	private static String BUTTON_CREATE_NETWORK = "Create network";
	private static String BUTTON_CREATE_PATTERNS = "Create patterns";
	private static String BUTTON_RUN = "Run";
	private static String BUTTON_SHOW_INPUTS = "Show inputs";
	private static String BUTTON_SHOW_WEIGHTS = "Show weights";
	private static String BUTTON_STOP = "Stop";

	private static JFrame frame;
	private static String INPUT_PATTERNS[] = { "Uniform distribution", "Uniform distribution: 10", "Uniform distribution: 100", "Circle", "Circle small", "Triangle" };
	private KohonenDrawingArea drawingArea;
	private JComboBox inputMenu;
	private int inputPatternType;
	private JLabel message;
	private JTextField neighMax, neighMin, etaMax, etaMin, numIterations, numNeuronsX, numNeuronsY;
	private Network network;
	private boolean running = false;
	private JButton showWeightsButton, runButton, showInputButton, createNetworkButton, stopButton, createPatternsButton;

	//-------------------------------------------------------------------------
	// Static methods
	//-------------------------------------------------------------------------
	
	/**
	 * Create the GUI and show it. For thread safety, this method should be
	 * invoked from the event-dispatching thread.
	 */
	private static void createAndShowGUI() {
		//Make sure we have nice window decorations.
		JFrame.setDefaultLookAndFeelDecorated(true);

		//Create and set up the window.
		frame = new JFrame("Kohonen Neural network Demo");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		//Set up the content pane.
		KohonenNeuNetDemo controller = new KohonenNeuNetDemo();
		controller.buildUserInterface(frame.getContentPane());

		// Create a neural network
		controller.buildNetwork();

		//Display the window.
		frame.pack();
		frame.setVisible(true);
	}

	/**
	 * Main 
	 * @param args
	 */
	public static void main(String[] args) {
		//Schedule a job for the event-dispatching thread:
		//creating and showing this application's GUI.
		javax.swing.SwingUtilities.invokeLater(new Runnable() {

			public void run() {
				createAndShowGUI();
			}
		});
	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------

	/**
	 * Action dispatcher
	 */
	public void actionPerformed(ActionEvent e) {
		if( e.getActionCommand().equals(BUTTON_RUN) ) {
			if( isRunning() ) {
				JOptionPane.showMessageDialog(frame, "Network already running!");
			} else {
				setRunning(true);
				message.setText("Network running");
				buildInputs(false);
				drawingArea.runNetwork();
			}
		} else if( e.getActionCommand().equals(BUTTON_SHOW_INPUTS) ) {
			showInputs(true);
		} else if( e.getActionCommand().equals(BUTTON_SHOW_WEIGHTS) ) {
			showWeights(true);
		} else if( e.getActionCommand().equals(BUTTON_STOP) ) {
			setRunning(false);
			message.setText("Network stopped");
		} else if( e.getActionCommand().equals(BUTTON_CREATE_NETWORK) ) {
			if( isRunning() ) {
				JOptionPane.showMessageDialog(frame, "Network already running. Stop it first");
			} else {
				network = null;
				buildNetwork();
				clear();
				showWeights(false);
			}
		} else if( e.getActionCommand().equals(BUTTON_CREATE_PATTERNS) ) {
			if( isRunning() ) {
				JOptionPane.showMessageDialog(frame, "Network already running. Stop it first");
			} else {
				clear();
				buildInputs(true);
				showInputs(true);
			}
		}
	}

	/**
	 * Find and set inputPatternType based on menu item
	 */
	public void buildInputPatternType() {
		// Set inputPatternType
		for( int i = 0; i < INPUT_PATTERNS.length; i++ ) {
			if( INPUT_PATTERNS[i].equals(inputMenu.getSelectedItem()) ) {
				inputPatternType = i;
				break;
			}
		}
	}

	/**
	 * Build an input pattern set
	 */
	public void buildInputs(boolean forceRebuild) {
		boolean rebuild = false;
		int oldIpt = inputPatternType;
		buildInputPatternType();

		// Do we need to rebuild?
		if( oldIpt != inputPatternType ) {
			if( (inputPatternType == 1) || (inputPatternType == 2) ) rebuild = true;
			if( (oldIpt == 1) || (oldIpt == 2) ) rebuild = true;
		}

		if( (network.getInput() == null) && ((inputPatternType == 1) || (inputPatternType == 2)) ) rebuild = true;

		// If no rebuild is nesceary, return
		if( !rebuild && !forceRebuild ) return;

		// Reset inputs
		network.setInput(null);

		// Create inputs (if needed)
		if( (inputPatternType == 1) || (inputPatternType == 2) ) {
			// Generate inputs
			int n, i, j, max = 10;
			if( inputPatternType == 2 ) max = 100;
			double input[][][] = new double[max][2][1];

			for( n = 0; n < max; n++ ) {
				for( i = 0; i < 2; i++ ) {
					for( j = 0; j < 1; j++ ) {
						input[n][i][j] = Math.random();
					}
				}
			}
			network.setInput(input);
		}

		setMessage("Input patterns created.");
	}

	/**
	 * Build a neural network
	 */
	public void buildNetwork() {
		if( network == null ) {
			// Create a network
			KohonenNetwork net = new KohonenNetwork(2);
			network = net;

			// Add input and output layers
			KohonenLayer klayIn = new KohonenLayer(2, 1, 0, 0, 0);
			network.addLayer(0, klayIn);
			KohonenLayer klayOut = new KohonenLayer(getNumNeuronsX(), getNumNeuronsY(), 2, 0, 1.0);
			network.addLayer(1, klayOut);

			// Connect neurons
			klayOut.connectAll(klayIn, false);
			buildInputs(false);
			setMessage("Network created. Layer size: [" + getNumNeuronsX() + ", " + getNumNeuronsY() + "]");
		}
	}

	/**
	 * Builds User Interface
	 * 
	 * @param container
	 */
	private void buildUserInterface(Container container) {
		int rowNum = 0;
		int minDimx = 120;
		int minDimy = 20;

		container.setLayout(new GridBagLayout());
		GridBagConstraints gbc = new GridBagConstraints();
		gbc.fill = GridBagConstraints.BOTH;

		gbc.weightx = 1;
		gbc.weighty = 1;

		// Add an area to draw
		drawingArea = new KohonenDrawingArea(this);
		gbc.gridx = 0;
		gbc.gridy = rowNum;
		gbc.gridwidth = 10;
		gbc.gridheight = 4;
		container.add(drawingArea, gbc);
		rowNum += 10;

		gbc.fill = GridBagConstraints.BOTH;
		gbc.weightx = 0;
		gbc.weighty = 0;

		// Add a 'message' label
		message = new JLabel(); //, drawingArea, JLabel.LEFT );
		message.setText("Ok");
		gbc.gridx = 0;
		gbc.gridy = rowNum++;
		gbc.gridwidth = 4;
		gbc.gridheight = 1;
		container.add(message, gbc);

		// Number neurons (layer's X size)
		numNeuronsX = new JTextField(2);
		numNeuronsX.setText("20");
		numNeuronsX.setMinimumSize(new Dimension(minDimx, minDimy));
		JLabel numNeuronsXLabel = new JLabel("Size X:", SwingConstants.RIGHT);
		numNeuronsXLabel.setMinimumSize(new Dimension(minDimx, minDimy));
		gbc.gridx = 0;
		gbc.gridy = rowNum;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(numNeuronsXLabel, gbc);
		gbc.gridx = 1;
		gbc.gridy = rowNum;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(numNeuronsX, gbc);
		numNeuronsXLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

		// Number neurons (layer's Y size)
		numNeuronsY = new JTextField(2);
		numNeuronsY.setText("20");
		numNeuronsY.setMinimumSize(new Dimension(minDimx, minDimy));
		JLabel numNeuronsYLabel = new JLabel("Size Y:", SwingConstants.RIGHT);
		numNeuronsYLabel.setMinimumSize(new Dimension(minDimx, minDimy));
		gbc.gridx = 2;
		gbc.gridy = rowNum;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(numNeuronsYLabel, gbc);
		gbc.gridx = 3;
		gbc.gridy = rowNum++;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(numNeuronsY, gbc);
		numNeuronsYLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

		// Add a button
		createNetworkButton = new JButton(BUTTON_CREATE_NETWORK);
		createNetworkButton.setMnemonic(KeyEvent.VK_C);
		createNetworkButton.setMinimumSize(new Dimension(minDimx, minDimy));
		gbc.gridx = 0;
		gbc.gridy = rowNum;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(createNetworkButton, gbc);
		createNetworkButton.addActionListener(this);

		// Add a button
		showWeightsButton = new JButton(BUTTON_SHOW_WEIGHTS);
		showWeightsButton.setMinimumSize(new Dimension(minDimx, minDimy));
		showWeightsButton.setMnemonic(KeyEvent.VK_W);
		gbc.gridx = 1;
		gbc.gridy = rowNum++;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(showWeightsButton, gbc);
		showWeightsButton.addActionListener(this);

		// Input patterns menu
		inputMenu = new JComboBox(INPUT_PATTERNS);
		gbc.gridx = 0;
		gbc.gridy = rowNum;
		gbc.gridwidth = 2;
		gbc.gridheight = 1;
		container.add(inputMenu, gbc);

		// Add a button
		createPatternsButton = new JButton(BUTTON_CREATE_PATTERNS);
		createPatternsButton.setMnemonic(KeyEvent.VK_C);
		createPatternsButton.setMinimumSize(new Dimension(minDimx, minDimy));
		gbc.gridx = 2;
		gbc.gridy = rowNum;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(createPatternsButton, gbc);
		createPatternsButton.addActionListener(this);

		// Add a button
		showInputButton = new JButton(BUTTON_SHOW_INPUTS);
		showInputButton.setMinimumSize(new Dimension(minDimx, minDimy));
		showInputButton.setMnemonic(KeyEvent.VK_I);
		gbc.gridx = 3;
		gbc.gridy = rowNum++;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(showInputButton, gbc);
		showInputButton.addActionListener(this);

		// Eta max
		etaMax = new JTextField(2);
		etaMax.setText("0.01");
		etaMax.setMinimumSize(new Dimension(minDimx, minDimy));
		JLabel etaMaxLabel = new JLabel("Eta max:", SwingConstants.RIGHT);
		etaMaxLabel.setMinimumSize(new Dimension(minDimx, minDimy));
		gbc.gridx = 0;
		gbc.gridy = rowNum;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(etaMaxLabel, gbc);
		gbc.gridx = 1;
		gbc.gridy = rowNum;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(etaMax, gbc);
		etaMaxLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

		// Eta min
		etaMin = new JTextField(2);
		etaMin.setText("0.01");
		etaMin.setMinimumSize(new Dimension(minDimx, minDimy));
		JLabel etaMinLabel = new JLabel("Eta min:", SwingConstants.RIGHT);
		etaMinLabel.setMinimumSize(new Dimension(minDimx, minDimy));
		gbc.gridx = 2;
		gbc.gridy = rowNum;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(etaMinLabel, gbc);
		gbc.gridx = 3;
		gbc.gridy = rowNum++;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(etaMin, gbc);
		etaMinLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

		// neigh max
		neighMax = new JTextField(2);
		neighMax.setText("15");
		neighMax.setMinimumSize(new Dimension(minDimx, minDimy));
		JLabel neighMaxLabel = new JLabel("Neighborhood max:", SwingConstants.RIGHT);
		neighMaxLabel.setMinimumSize(new Dimension(minDimx, minDimy));
		gbc.gridx = 0;
		gbc.gridy = rowNum;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(neighMaxLabel, gbc);
		gbc.gridx = 1;
		gbc.gridy = rowNum;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(neighMax, gbc);
		neighMaxLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

		// neigh min
		neighMin = new JTextField(2);
		neighMin.setText("0");
		neighMin.setMinimumSize(new Dimension(minDimx, minDimy));
		JLabel neighMinLabel = new JLabel("Neighborhood min:", SwingConstants.RIGHT);
		neighMinLabel.setMinimumSize(new Dimension(minDimx, minDimy));
		gbc.gridx = 2;
		gbc.gridy = rowNum;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(neighMinLabel, gbc);
		gbc.gridx = 3;
		gbc.gridy = rowNum++;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(neighMin, gbc);
		neighMinLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

		// Number of iterations
		numIterations = new JTextField(2);
		numIterations.setText("10000");
		numIterations.setMinimumSize(new Dimension(minDimx, minDimy));
		JLabel numIterationsLabel = new JLabel("Iterations:", SwingConstants.RIGHT);
		numIterationsLabel.setMinimumSize(new Dimension(minDimx, minDimy));
		gbc.gridx = 0;
		gbc.gridy = rowNum;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(numIterationsLabel, gbc);
		gbc.gridx = 1;
		gbc.gridy = rowNum++;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(numIterations, gbc);
		numIterationsLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

		// Add a button
		runButton = new JButton(BUTTON_RUN);
		runButton.setMinimumSize(new Dimension(minDimx, minDimy));
		runButton.setMnemonic(KeyEvent.VK_R);
		gbc.gridx = 0;
		gbc.gridy = rowNum;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(runButton, gbc);
		runButton.addActionListener(this);

		// Add a button
		stopButton = new JButton(BUTTON_STOP);
		stopButton.setMnemonic(KeyEvent.VK_S);
		stopButton.setMinimumSize(new Dimension(minDimx, minDimy));
		gbc.gridx = 1;
		gbc.gridy = rowNum++;
		gbc.gridwidth = 1;
		gbc.gridheight = 1;
		container.add(stopButton, gbc);
		stopButton.addActionListener(this);

	}

	public void clear() {
		drawingArea.clear();
	}

	public double getEtaMax() {
		try {
			return Double.parseDouble(etaMax.getText());
		} catch(Exception e) {
			return 0;
		}
	}

	public double getEtaMin() {
		try {
			return Double.parseDouble(etaMin.getText());
		} catch(Exception e) {
			return 0;
		}
	}

	public int getInputPatternType() {
		return inputPatternType;
	}

	public double getNeighMax() {
		try {
			return Double.parseDouble(neighMax.getText());
		} catch(Exception e) {
			return 0;
		}
	}

	public double getNeighMin() {
		try {
			return Double.parseDouble(neighMin.getText());
		} catch(Exception e) {
			return 0;
		}
	}

	public Network getNetwork() {
		return network;
	}

	public int getNumIterations() {
		try {
			return Integer.parseInt(numIterations.getText());
		} catch(Exception e) {
			return 0;
		}
	}

	public int getNumNeuronsX() {
		try {
			return Integer.parseInt(numNeuronsX.getText());
		} catch(Exception e) {
			return 0;
		}
	}

	public int getNumNeuronsY() {
		try {
			return Integer.parseInt(numNeuronsY.getText());
		} catch(Exception e) {
			return 0;
		}
	}

	public boolean isRunning() {
		return running;
	}

	/**
	 * Select an input pattern (randomly) and set input layer
	 */
	void setInputPattern() {
		double x = 0, y = 0;
		if( getInputPatternType() == 0 ) {
			x = Math.random();
			y = Math.random();
		} else if( getInputPatternType() == 1 ) {
			int r = (int) (Math.random() * 10);
			x = network.getInput()[r][0][0];
			y = network.getInput()[r][1][0];
		} else if( getInputPatternType() == 2 ) {
			int r = (int) (Math.random() * 100);
			x = network.getInput()[r][0][0];
			y = network.getInput()[r][1][0];
		} else if( getInputPatternType() == 3 ) {
			double r = 2.0;
			double rmax = 0.5;
			double rr = rmax * rmax;
			while(r > rr) {
				x = (2 * Math.random() - 1) * rmax;
				y = (2 * Math.random() - 1) * rmax;
				r = x * x + y * y;
			}
			x += 0.5;
			y += 0.5;
		} else if( getInputPatternType() == 4 ) {
			double r = 2.0;
			double rmax = 0.25;
			double rr = rmax * rmax;
			while(r > rr) {
				x = (2 * Math.random() - 1) * rmax;
				y = (2 * Math.random() - 1) * rmax;
				r = x * x + y * y;
			}
			x += 0.5;
			y += 0.5;
		} else if( getInputPatternType() == 5 ) {
			x = Math.random();
			y = Math.random() * x;
			y = 1 - y;
		} else {
			throw new RuntimeException("Unknown pattern type!");
		}
		network.getLayer(0).getNeuron(0, 0).setOutput(x);
		network.getLayer(0).getNeuron(1, 0).setOutput(y);
	}

	public void setMessage(String text) {
		message.setText(text);
	}

	public void setRunning(boolean running) {
		this.running = running;
	}

	public void showInputs(boolean showLablel) {
		drawingArea.showInputs();
		if( showLablel ) setMessage("Input patterns: " + (network.getInput() == null ? "Inf." : Integer.toString(network.getInput().length)));
	}

	public void showWeights(boolean showLablel) {
		drawingArea.showWeights();
		if( showLablel ) setMessage("Layer size: [" + network.getLayer(1).getSizeX() + ", " + network.getLayer(1).getSizeX() + "]");
	}
}

